import hashlib
import io
import re
from datetime import datetime
from typing import List, Tuple, Optional

import requests
import pdfplumber
from bs4 import BeautifulSoup
import snowflake.connector

from snowflake_config import SF_CONFIG


WPSR_HOME = "https://www.eia.gov/petroleum/supply/weekly/"

# 用来识别 Highlights 页的 6 个锚点（小写以后匹配）
HIGHLIGHTS_ANCHORS = [
    "u.s. crude oil refinery inputs averaged",
    "u.s. crude oil imports averaged",
    "u.s. commercial crude oil inventories",
    "total products supplied over the",
    "the price for west texas intermediate crude oil",
    "the national average retail price for",
]


# ---------- 工具函数 ----------

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def fetch_html(url: str) -> str:
    print(f"[HTTP] GET {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def extract_week_ending_and_release_date(html: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    week_ending = None
    release_dt = None

    m1 = re.search(r"Data for week ending\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
    if m1:
        week_ending = datetime.strptime(m1.group(1), "%B %d, %Y")

    m2 = re.search(r"Release Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
    if m2:
        release_dt = datetime.strptime(m2.group(1), "%B %d, %Y")

    return week_ending, release_dt


def get_latest_archive_urls(n_issues: int = 8) -> List[str]:
    html = fetch_html(WPSR_HOME)
    soup = BeautifulSoup(html, "html.parser")

    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if (
            "/petroleum/supply/weekly/archive/" in href
            and "wpsr_" in href
            and href.endswith(".php")
        ):
            full_url = href
            if not full_url.startswith("http"):
                full_url = "https://www.eia.gov" + full_url
            urls.append(full_url)

    seen = set()
    dedup = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)

    print(f"[INFO] Found {len(dedup)} archive URLs, using first {n_issues}.")
    return dedup[:n_issues]


def archive_to_full_pdf_url(archive_url: str) -> str:
    base = archive_url.rsplit("/", 1)[0]
    return f"{base}/pdf/wpsrall.pdf"


# ---------- PDF Highlights Extract ----------

def extract_highlights_from_pdf(pdf_bytes: bytes) -> Optional[str]:
    """
    从 wpsrall.pdf 中提取 Highlights 页文本：

      1）遍历每一页，基于“Highlights + 6 个锚点句子”打分，选择得分最高的页面；
      2）在该页上用 extract_words() 取出所有单词的 x 中心点，找出 x 上的“最大空隙”，
         其中心作为列分界 split_x；
      3）用 page.crop((0, 0, split_x, h)) / page.crop((split_x, 0, w, h))
         分别提取左/右列文本；
      4）正常情况下返回 left + '\\n\\n\\n\\n' + right；
         任一步失败则退回整页 extract_text() 结果。

    目标是：
      - 不再出现“整期 Highlights 跳过，Bronze 表里没数据”的情况；
      - 不在我们自己逻辑里裁掉列边缘的单词，右上角那句 inventories 也能保留。
    """

    def split_page_two_columns(page) -> str:
        """在给定 page 上尝试按列拆分，失败时抛出异常，由上层兜底整页文本。"""
        words = page.extract_words()
        if not words:
            raise RuntimeError("no words on page")

        xs = [(w["x0"] + w["x1"]) / 2.0 for w in words]
        if not xs:
            raise RuntimeError("no x centers on page")

        min_x = min(xs)
        max_x = max(xs)
        if max_x - min_x < 40:
            # 看上去不是双栏，直接整页
            raise RuntimeError("page not wide enough for two columns")

        xs_sorted = sorted(xs)
        gaps = [(xs_sorted[i + 1] - xs_sorted[i], i) for i in range(len(xs_sorted) - 1)]
        max_gap, idx = max(gaps, key=lambda g: g[0])
        split_x = (xs_sorted[idx] + xs_sorted[idx + 1]) / 2.0

        page_width = float(page.width)
        page_height = float(page.height)

        margin = 5.0
        split_x = max(margin, min(page_width - margin, split_x))

        left_crop = page.crop((0, 0, split_x, page_height))
        right_crop = page.crop((split_x, 0, page_width, page_height))

        left_text = (left_crop.extract_text() or "").strip()
        right_text = (right_crop.extract_text() or "").strip()

        if not left_text and not right_text:
            raise RuntimeError("both left and right text empty")

        if left_text and right_text:
            return left_text + "\n\n\n\n" + right_text
        else:
            # 至少有一列有东西，也先返回，不再死抠列结构
            return (left_text or right_text)

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        best_idx = None
        best_score = -1
        page_texts = []

        # 1) 先选出“最像 Highlights 的页面”
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            page_texts.append(txt)
            lower = txt.lower()
            if not lower.strip():
                continue

            score = 0
            if "highlights" in lower:
                score += 2
            for ph in HIGHLIGHTS_ANCHORS:
                if ph in lower:
                    score += 1

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            # 极端情况：整份 PDF 都没 extract_text，放弃
            return None

        print(f"[DEBUG] Selected page #{best_idx} as Highlights (score={best_score}).")

        page = pdf.pages[best_idx]
        base_txt = page_texts[best_idx] or (page.extract_text() or "")

        # 2) 尝试按列拆分，失败时退回整页 base_txt
        try:
            col_text = split_page_two_columns(page)
            col_text = col_text.strip()
            if col_text:
                return col_text
        except Exception as e:
            print(f"[WARN] column split failed on page {best_idx}: {e}")

        base_txt = (base_txt or "").strip()
        return base_txt if base_txt else None


# ---------- Snowflake ----------

def sf_connect():
    return snowflake.connector.connect(
        user=SF_CONFIG["user"],
        password=SF_CONFIG["password"],
        account=SF_CONFIG["account"],
        warehouse=SF_CONFIG["warehouse"],
        database=SF_CONFIG["database"],
        schema=SF_CONFIG["schema"],
        role=SF_CONFIG["role"],
    )


def ensure_bronze_highlights_table(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS BRONZE_RAW_DOCUMENT_HIGHLIGHTS (
        document_id   STRING,
        source_system STRING,
        doc_type      STRING,
        title         STRING,
        week_ending   DATE,
        published_at  TIMESTAMP_NTZ,
        ingested_at   TIMESTAMP_NTZ,
        raw_uri       STRING,
        raw_text      STRING,
        raw_hash      STRING
    );
    """
    cur = conn.cursor()
    try:
        cur.execute(sql)
        print("[INIT] Ensured BRONZE_RAW_DOCUMENT_HIGHLIGHTS exists.")
    finally:
        cur.close()


def upsert_bronze_highlights(conn, row: dict):
    delete_sql = """
        DELETE FROM BRONZE_RAW_DOCUMENT_HIGHLIGHTS
        WHERE document_id = %(document_id)s;
    """
    insert_sql = """
        INSERT INTO BRONZE_RAW_DOCUMENT_HIGHLIGHTS (
            document_id, source_system, doc_type, title,
            week_ending, published_at, ingested_at,
            raw_uri, raw_text, raw_hash
        )
        VALUES (
            %(document_id)s, %(source_system)s, %(doc_type)s, %(title)s,
            %(week_ending)s, %(published_at)s, %(ingested_at)s,
            %(raw_uri)s, %(raw_text)s, %(raw_hash)s
        );
    """
    cur = conn.cursor()
    try:
        cur.execute(delete_sql, row)
        cur.execute(insert_sql, row)
    finally:
        cur.close()


# ---------- 主流程 ----------

def ingest_latest_wpsr_highlights(n_issues: int = 12):
    archive_urls = get_latest_archive_urls(n_issues=n_issues)

    conn = sf_connect()
    try:
        ensure_bronze_highlights_table(conn)

        for aurl in archive_urls:
            print(f"\n[ISSUE] Processing archive: {aurl}")
            ahtml = fetch_html(aurl)
            week_ending_dt, release_dt = extract_week_ending_and_release_date(ahtml)

            pdf_url = archive_to_full_pdf_url(aurl)
            print(f"[PDF] {pdf_url}")
            pdf_resp = requests.get(pdf_url, timeout=120)
            pdf_resp.raise_for_status()

            highlights_text = extract_highlights_from_pdf(pdf_resp.content)

            if not highlights_text:
                print("[WARN] No Highlights found, skip.")
                continue

            if week_ending_dt:
                document_id = f"EIA_WPSR_{week_ending_dt.date().isoformat()}"
                title = f"EIA WPSR Highlights - Week Ending {week_ending_dt.date().isoformat()}"
                week_ending_str = week_ending_dt.date().isoformat()
            else:
                document_id = f"EIA_WPSR_{sha256_text(aurl)[:10]}"
                title = "EIA WPSR Highlights (Unknown Week Ending)"
                week_ending_str = None

            now = datetime.utcnow()

            row_hl = dict(
                document_id=document_id,
                source_system="EIA_WPSR",
                doc_type="report_pdf_highlights",
                title=title,
                week_ending=week_ending_str,
                published_at=release_dt,
                ingested_at=now,
                raw_uri=pdf_url,
                raw_text=highlights_text,
                raw_hash=sha256_text(highlights_text),
            )

            upsert_bronze_highlights(conn, row_hl)
            print(f"[OK] Upserted HIGHLIGHTS document_id={document_id}")

    finally:
        conn.close()


if __name__ == "__main__":
    ingest_latest_wpsr_highlights(n_issues=8)
