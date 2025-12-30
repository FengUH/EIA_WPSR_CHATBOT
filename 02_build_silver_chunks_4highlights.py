"""
02_build_silver_chunks_4highlights.py

从 BRONZE_RAW_DOCUMENT_HIGHLIGHTS 读取 Highlights 页的原始文本 raw_text：
  1）轻清洗：去掉页眉 "High"/"Highlights"；
  2）利用 Bronze 插入的分隔符 "\n\n\n\n" 把左列 / 右列拆开；
  3）对每一列分别在表格标题前截断（去掉 Refinery Activity / Stocks / Net Imports /
     Products Supplied / Prices 等表格和脚注）；
  4）再把两列拼回一个文本；
  5）基于 6 个段落开头（anchor phrases）做容错匹配，按锚点顺序切成段；
  6）对每个段落做行内重排 + 仅做结构性噪声清理（不改动语义内容）；
  7）写入 SILVER_CHUNK_HIGHLIGHTS 表，对每个 document_id 先 DELETE 再 INSERT（幂等）。
"""

from datetime import datetime
from typing import List, Dict, Optional, Tuple
import re

import snowflake.connector
from snowflake_config import SF_CONFIG


BRONZE_TABLE_NAME = "BRONZE_RAW_DOCUMENT_HIGHLIGHTS"
SILVER_TABLE_NAME = "SILVER_CHUNK_HIGHLIGHTS"


# ========== 1. 文本预处理：左右列 + 去表格 ==========

ANCHOR_PHRASES = [
    "U.S. crude oil refinery inputs averaged",
    "U.S. crude oil imports averaged",
    "U.S. commercial crude oil inventories",
    "Total products supplied over the",
    "The price for West Texas Intermediate crude oil",
    "The national average retail price for",
]

END_PATTERN = re.compile(
    r"(Refinery Activity\s*\(Thousand Barrels per Day\)"
      r"|Stocks\s*\(Million Barrels\)"
      r"|Net Imports\s*\(Thousand Barrels per Day\)"
      r"|Products Supplied\s*\(Thousand Barrels per Day\)"
      r"|Prices\s*\(Dollars per Gallon except as noted\))",
    re.IGNORECASE | re.DOTALL,
)


def _truncate_tables(col_text: str) -> str:
    """在单列文本中，从第一个表格标题开始截断，去掉下面所有表格+脚注。"""
    if not col_text:
        return col_text
    m = END_PATTERN.search(col_text)
    if m:
        col_text = col_text[: m.start()]
    return col_text.rstrip()


def preprocess_highlights_text(raw_text: str) -> Optional[str]:
    """
    针对 BRONZE.raw_text 做清洗：
      - 替换 \r 为 \n；
      - 去掉顶部单独的 "High"/"Highlights" 页眉行；
      - 利用 "\n\n\n\n" 把左列 / 右列拆开，对每一列单独做表格截断；
      - 再把两列用同样的分隔符拼回去。
    """
    if not raw_text:
        return None

    text = raw_text.replace("\r", "\n")

    # 去掉顶部的 "High" / "Highlights" 页眉行
    lines = text.splitlines()
    while lines and lines[0].strip().lower() in ("high", "highlights"):
        lines = lines[1:]
    text = "\n".join(lines).lstrip()

    # 利用 Bronze 插入的分隔符，把左/右列拆开
    if "\n\n\n\n" in text:
        left, right = text.split("\n\n\n\n", 1)
        left = _truncate_tables(left.strip())
        right = _truncate_tables(right.strip())
        parts = [p for p in (left, right) if p]
        text = "\n\n\n\n".join(parts)
    else:
        # 极端情况：没有明显两列分隔符，当作单列处理
        text = _truncate_tables(text)

    text = text.strip()
    if not text:
        return None
    return text


def _build_pattern(phrase: str) -> re.Pattern:
    """anchor phrase -> 容错正则（允许中间有任意空白，大小写不敏感）。"""
    words = phrase.split()
    pattern = r"\s+".join(re.escape(w) for w in words)
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


ANCHOR_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (phrase, _build_pattern(phrase)) for phrase in ANCHOR_PHRASES
]


def _find_anchor_positions(text: str) -> List[Tuple[str, int]]:
    """返回 [(phrase, start_index), ...]，按出现顺序排序。"""
    positions: List[Tuple[str, int]] = []
    for phrase, pattern in ANCHOR_PATTERNS:
        m = pattern.search(text)
        if m:
            positions.append((phrase, m.start()))
    positions.sort(key=lambda x: x[1])
    return positions


def build_highlights_paragraph_chunks(text: str) -> List[Dict]:
    """
    基于 6 个锚点切段：
      - 每个锚点起始位置到下一个锚点起始位置之间 => 一个段落；
      - 最后一个锚点到文本末尾 => 最后一个段落；
      - 如果一个锚点都没找到 => 整页作为一个 chunk。
    """
    if not text:
        return []

    anchor_positions = _find_anchor_positions(text)

    if not anchor_positions:
        t = text.strip()
        if not t:
            return []
        return [dict(chunk_index=1, text=t, char_start=0, char_end=len(text) - 1)]

    chunks: List[Dict] = []
    for i, (_, start_idx) in enumerate(anchor_positions):
        cur_start = start_idx
        if i + 1 < len(anchor_positions):
            next_start = anchor_positions[i + 1][1]
        else:
            next_start = len(text)

        raw_para = text[cur_start:next_start]

        leading_ws = len(raw_para) - len(raw_para.lstrip())
        trailing_ws = len(raw_para) - len(raw_para.rstrip())

        para_clean = raw_para.strip()
        if not para_clean:
            continue

        char_start = cur_start + leading_ws
        char_end = next_start - 1 - trailing_ws

        chunks.append(
            dict(
                chunk_index=len(chunks) + 1,
                text=para_clean,
                char_start=char_start,
                char_end=char_end,
            )
        )

    return chunks


# ========== 2. 段落级清理（结构性） ==========

def clean_chunk_text(text: str) -> str:
    """
    针对单个段落做 **结构性** 清理（不改语义）：
      - 把所有换行折叠成空格；
      - 连续空白折叠成一个空格；
      - 用占位符保护各种写法的 U.S.（U . S . / U . S / U S / U.S.）；
      - 把 "word . word" 缩成 "word word"（去掉游离句号）；
      - 删除所有孤立单字母词（除 a/A/I）；
      - 删除孤立的残词 "hlights"；
      - 删除尾部多余的 "Highlights ..."；
      - 合并结尾连续标点（", .", ".,," 等）为一个 "."；
      - 最后把占位符还原为 "U.S."。
    """
    if not text:
        return ""

    # 统一成一行
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 占位符保护 U.S.（包括 U . S . / U . S / U S / U.S.）
    US_PLACEHOLDER = "__US_PLACEHOLDER__"
    text = re.sub(
        r"\bU\s*\.?\s*S\s*\.?(?=\s|$)",
        US_PLACEHOLDER,
        text,
        flags=re.IGNORECASE,
    )

    # 把 "word . word" 这种多余句号变成 "word word"
    text = re.sub(
        r"(\w)\s+\.\s+(\w)",
        r"\1 \2",
        text,
    )

    # 删除所有孤立单字母词（除了 a/A/I）
    text = re.sub(
        r"\b(?!(?:[Aa]|I)\b)[A-Za-z]\b",
        " ",
        text,
    )

    # 删除孤立残词 "hlights"
    text = re.sub(
        r"\bhlights\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # 再折叠一次空白
    text = re.sub(r"\s+", " ", text).strip()

    # 兜底：如果还有乱入的 "Highlights ..."，直接砍掉后半截
    text = re.sub(r"\bHighlights\b.*$", "", text, flags=re.IGNORECASE)

    # 合并多重句点
    text = re.sub(r"\.{2,}", ".", text)

    # 把结尾的连续标点（.,;: 等组合）缩成一个句号
    text = re.sub(r"[,\.;:]+$", ".", text)

    # 还原 U.S.
    text = text.replace(US_PLACEHOLDER, "U.S.")

    return text.strip()


def infer_topics(text: str) -> Optional[str]:
    """非常粗糙的关键词打标。"""
    t = text.lower()
    topics = []

    if "crude" in t:
        topics.append("crude_oil")
    if "gasoline" in t:
        topics.append("gasoline")
    if "distillate" in t:
        topics.append("distillate")
    if "propane" in t:
        topics.append("propane")
    if "refinery" in t or "utilization" in t:
        topics.append("refinery")
    if "inventory" in t or "stock" in t:
        topics.append("inventory")

    if not topics:
        return None
    topics = sorted(set(topics))
    return ",".join(topics)


# ========== 3. Snowflake 相关 ==========

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


def ensure_silver_highlights_table(conn):
    sql = f"""
    CREATE TABLE IF NOT EXISTS {SILVER_TABLE_NAME} (
        chunk_id      STRING,
        document_id   STRING,
        chunk_index   NUMBER,
        chunk_text    STRING,
        char_start    NUMBER,
        char_end      NUMBER,
        chunk_type    STRING,      -- 'highlights'
        week_ending   DATE,
        published_at  TIMESTAMP_NTZ,
        topics        STRING,
        created_at    TIMESTAMP_NTZ
    );
    """
    cur = conn.cursor()
    try:
        cur.execute(sql)
        print(f"[INIT] Ensured table {SILVER_TABLE_NAME} exists.")
    finally:
        cur.close()


def fetch_all_bronze_docs(conn) -> List[Dict]:
    sql = f"""
    SELECT
      document_id,
      week_ending,
      published_at,
      raw_text
    FROM {BRONZE_TABLE_NAME}
    ORDER BY week_ending;
    """
    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [c[0].lower() for c in cur.description]
    finally:
        cur.close()

    docs = []
    for r in rows:
        docs.append(dict(zip(cols, r)))
    return docs


def delete_chunks_for_document(conn, document_id: str):
    sql = f"DELETE FROM {SILVER_TABLE_NAME} WHERE document_id = %(document_id)s"
    cur = conn.cursor()
    try:
        cur.execute(sql, {"document_id": document_id})
    finally:
        cur.close()


def insert_chunks(conn, records: List[Dict]):
    if not records:
        return
    sql = f"""
    INSERT INTO {SILVER_TABLE_NAME} (
        chunk_id,
        document_id,
        chunk_index,
        chunk_text,
        char_start,
        char_end,
        chunk_type,
        week_ending,
        published_at,
        topics,
        created_at
    ) VALUES (
        %(chunk_id)s,
        %(document_id)s,
        %(chunk_index)s,
        %(chunk_text)s,
        %(char_start)s,
        %(char_end)s,
        %(chunk_type)s,
        %(week_ending)s,
        %(published_at)s,
        %(topics)s,
        %(created_at)s
    );
    """
    cur = conn.cursor()
    try:
        cur.executemany(sql, records)
    finally:
        cur.close()


# ========== 4. 主流程 ==========

def main():
    conn = sf_connect()
    try:
        ensure_silver_highlights_table(conn)

        docs = fetch_all_bronze_docs(conn)
        print(f"[INFO] Loaded {len(docs)} documents from {BRONZE_TABLE_NAME}.")

        for doc in docs:
            doc_id = doc["document_id"]
            raw_text = doc["raw_text"]
            week_ending = doc["week_ending"]
            published_at = doc["published_at"]

            if not raw_text:
                print(f"[WARN] document_id={doc_id} has empty raw_text, skip.")
                continue

            core_text = preprocess_highlights_text(raw_text)
            if not core_text:
                print(f"[WARN] document_id={doc_id} has no valid preprocessed Highlights text, skip.")
                continue

            print(f"\n[DOC] Processing HIGHLIGHTS paragraph chunks for document_id={doc_id}")

            raw_chunks = build_highlights_paragraph_chunks(core_text)
            if not raw_chunks:
                print(f"[WARN] document_id={doc_id} produced 0 chunks, skip.")
                continue

            # 幂等：删除旧 chunks
            delete_chunks_for_document(conn, document_id=doc_id)

            now = datetime.utcnow()
            records: List[Dict] = []

            for c in raw_chunks:
                cleaned_text = clean_chunk_text(c["text"])
                if not cleaned_text:
                    continue

                topics_str = infer_topics(cleaned_text)
                chunk_id = f"{doc_id}_HL{c['chunk_index']:03d}"

                rec = dict(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    chunk_index=c["chunk_index"],
                    chunk_text=cleaned_text,
                    char_start=c["char_start"],
                    char_end=c["char_end"],
                    chunk_type="highlights",
                    week_ending=week_ending,
                    published_at=published_at,
                    topics=topics_str,
                    created_at=now,
                )
                records.append(rec)

            insert_chunks(conn, records)
            print(f"[OK] Inserted {len(records)} paragraph chunks for {doc_id}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
