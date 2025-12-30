"""
06_evaluate_highlights.py  (Benchmark v2)

自动评估：基于 EIA WPSR Highlights 的 RAG Chatbot 数字准确率（多周、多问题模板的核心版）

当前主要评估维度（针对 inventories 段 chunk_index=3）：
  - crude_change_phrase:   "decreased by 1.3 million barrels"
  - gasoline_change_phrase:"increased by 4.8 million barrels"
  - crude_avg_phrase:      "4% below the five year average"
  - gasoline_avg_phrase:   "slightly below the five year average" 等

流程：
1. 从 RAW.SILVER_CHUNK_HIGHLIGHTS 读取最近 N 周的 inventories 段（chunk_index=3）
2. 对每一周的段落用正则抽取 ground truth 短语
3. 为每一周生成一个明确指定周的提问：
   "In the week ending {week}, what happened to ... and how do they compare..."
4. 调用 05_answer_question_4highlights.py 中的检索 + Prompt + LLM 生成回答
5. 检查回答中是否包含正确的数字 / five-year average 描述
6. 输出逐周 summary 表格 + 各字段平均准确率
"""

import re
import json
import importlib.util
from pathlib import Path

import snowflake.connector
from snowflake_config import SF_CONFIG


# ------------ 动态加载你的 04_answer_question_4highlights.py --------------

SCRIPT_DIR = Path(__file__).resolve().parent
ANSWER_SCRIPT = SCRIPT_DIR / "04_answer_question_4highlights.py"

spec = importlib.util.spec_from_file_location("answer_hl", ANSWER_SCRIPT)
answer_hl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answer_hl)


# ------------ Snowflake 连接（用于读 SILVER 表） ------------

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


# ------------ 从 SILVER_CHUNK_HIGHLIGHTS 读最近 N 周 inventories 段 ------------

def load_inventory_paragraphs(conn, n_weeks: int = 6):
    """
    取最近 n_weeks 周的 chunk_index=3（inventories 段）作为 ground truth 来源。
    按 week_ending 从新到旧排序。
    """
    sql = """
    SELECT week_ending, chunk_text
    FROM RAW.SILVER_CHUNK_HIGHLIGHTS
    WHERE chunk_index = 3      -- 第 3 段：inventories
    ORDER BY week_ending DESC
    LIMIT %s
    """
    cur = conn.cursor()
    cur.execute(sql, (n_weeks,))
    rows = cur.fetchall()
    cur.close()

    docs = []
    for week_ending, text in rows:
        docs.append(dict(week=str(week_ending), text=text))
    return docs


# ------------ 从 inventories 段里抽 ground truth ------------

def extract_ground_truth(paragraph_text: str):
    """
    从清洗后的 inventories 段文本中抽取：
      - crude_change_phrase: "decreased by 1.3 million barrels"
      - gasoline_change_phrase: "increased by 4.8 million barrels"
      - crude_avg_phrase: crude inventories 那句里的 "X% above/below the five year average"
      - gasoline_avg_phrase: "slightly below the five year average" 或 "X% below/above the five year average"
    """

    # 1) 文本标准化：five-year / five  year -> five year
    t = re.sub(r"five[-\s]+year", "five year", paragraph_text, flags=re.IGNORECASE)

    # ----------------- crude 部分 -----------------
    crude_change_phrase = None
    crude_avg_phrase = None

    # crude change: 在整段里找 “U.S. commercial crude oil inventories … increased/decreased by X million barrels”
    m_change = re.search(
        r"U\.S\. commercial crude oil inventories.*?(increased|decreased) by ([\d\.]+) million barrels",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_change:
        crude_change_phrase = f"{m_change.group(1).lower()} by {m_change.group(2)} million barrels"

    # crude five-year: 在整段里找 “U.S. crude oil inventories … about X% above/below the five year average”
    m_crude_avg = re.search(
        r"U\.S\. crude oil inventories.*?about\s+([\d\.]+)%\s+(above|below)\s+the five year average",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_crude_avg:
        crude_avg_phrase = f"{m_crude_avg.group(1)}% {m_crude_avg.group(2).lower()} the five year average"

    # ----------------- gasoline 部分 -----------------
    gasoline_change_phrase = None
    gasoline_avg_phrase = None

    m_gas_sent = re.search(
        r"(Total motor gasoline inventories.*?\.)(?:\s|$)",
        t,
        flags=re.IGNORECASE,
    )
    if m_gas_sent:
        gas_sent = m_gas_sent.group(1)

        m_gas_change = re.search(
            r"(increased|decreased) by ([\d\.]+) million barrels",
            gas_sent,
            flags=re.IGNORECASE,
        )
        if m_gas_change:
            gasoline_change_phrase = f"{m_gas_change.group(1).lower()} by {m_gas_change.group(2)} million barrels"

        # five-year 对比（优先 slightly below，其次 about X% above/below）
        if re.search(r"slightly below the five year average", gas_sent, flags=re.IGNORECASE):
            gasoline_avg_phrase = "slightly below the five year average"
        else:
            m_gas_avg = re.search(
                r"about\s+([\d\.]+)%\s+(above|below)\s+the five year average",
                gas_sent,
                flags=re.IGNORECASE,
            )
            if m_gas_avg:
                gasoline_avg_phrase = f"{m_gas_avg.group(1)}% {m_gas_avg.group(2).lower()} the five year average"

    gt = dict(
        crude_change_phrase=crude_change_phrase,
        gasoline_change_phrase=gasoline_change_phrase,
        crude_avg_phrase=crude_avg_phrase,
        gasoline_avg_phrase=gasoline_avg_phrase,
    )
    return gt

# ------------ 让你的 chatbot 回答同样的问题 ------------

def get_bot_answer(question: str, top_k: int = 6, weeks_back: int = 12) -> str:
    """
    不直接调用 answer_question（它主要负责打印），
    而是复用 05 脚本里的底层函数来拿到回答字符串。
    """
    conn = answer_hl.sf_connect()
    try:
        chunks = answer_hl.retrieve_top_chunks(conn, question, top_k=top_k, weeks_back=weeks_back)
    finally:
        conn.close()

    prompt = answer_hl.build_prompt(question, chunks)
    answer = answer_hl.call_local_llm(prompt)
    return answer


# ------------ 评估回答是否包含正确数字 / 描述 ------------

def evaluate_answer(gt: dict, answer: str):
    """
    简单字符串匹配评估：
      - crude_change_phrase 是否出现在回答里
      - gasoline_change_phrase 是否出现在回答里
      - crude_avg_phrase 是否出现在回答里
      - gasoline_avg_phrase 是否出现在回答里

    返回 per-field bool + overall_accuracy（在有 ground truth 的字段上的平均）。
    """

    # normalize：大小写 + five-year / five year
    norm_answer = answer.lower().replace("five-year", "five year")

    def contains(phrase: str | None) -> bool:
        if not phrase:
            return False
        return phrase.lower() in norm_answer

    result = {
        "crude_change_ok": contains(gt.get("crude_change_phrase")),
        "gasoline_change_ok": contains(gt.get("gasoline_change_phrase")),
        "crude_avg_ok": contains(gt.get("crude_avg_phrase")),
        "gasoline_avg_ok": contains(gt.get("gasoline_avg_phrase")),
    }

    # overall accuracy：只在 ground truth 存在的字段上统计
    num_checks = sum(1 for k, v in gt.items() if v)
    num_correct = 0
    if gt.get("crude_change_phrase"):
        num_correct += int(result["crude_change_ok"])
    if gt.get("gasoline_change_phrase"):
        num_correct += int(result["gasoline_change_ok"])
    if gt.get("crude_avg_phrase"):
        num_correct += int(result["crude_avg_ok"])
    if gt.get("gasoline_avg_phrase"):
        num_correct += int(result["gasoline_avg_ok"])

    accuracy = num_correct / num_checks if num_checks else 0.0
    result["overall_accuracy"] = accuracy

    return result


# ------------ 主流程：多周、多问题 benchmark ------------

def main():
    conn = sf_connect()
    try:
        # 1) 取最近 N 周 inventories 段并抽 ground truth
        n_weeks = 6
        docs = load_inventory_paragraphs(conn, n_weeks=n_weeks)
        gt_list = []
        for d in docs:
            gt = extract_ground_truth(d["text"])
            gt_list.append(dict(week=d["week"], gt=gt))
    finally:
        conn.close()

    print("=== Ground truth for recent weeks (inventories paragraph) ===")
    for item in gt_list:
        print(f"Week {item['week']}:")
        print(json.dumps(item["gt"], indent=2))
    print()

    # 2) 对每一周生成一个问题 + 调用 RAG + 评估
    results_per_week = []

    # 用于聚合 accuracy
    agg_hits = {"crude_change_ok": 0, "gasoline_change_ok": 0, "crude_avg_ok": 0, "gasoline_avg_ok": 0}
    agg_denoms = {"crude_change_ok": 0, "gasoline_change_ok": 0, "crude_avg_ok": 0, "gasoline_avg_ok": 0}

    for item in gt_list:
        week = item["week"]
        gt = item["gt"]

        question = (
            f"In the week ending {week}, what happened to U.S. commercial crude oil inventories "
            f"and motor gasoline inventories, and how do they compare with their five-year averages?"
        )

        print(f"\n=== Asking bot for week {week} ===")
        print(f"Question: {question}\n")

        answer = get_bot_answer(question)

        print("BOT ANSWER:")
        print(answer)
        print()

        eval_result = evaluate_answer(gt, answer)

        # 对于没有 ground truth 的字段，用 None 标记，在表里打印成 "-"
        results_per_week.append(
            dict(
                week=week,
                crude_change_ok=eval_result["crude_change_ok"] if gt.get("crude_change_phrase") else None,
                gasoline_change_ok=eval_result["gasoline_change_ok"] if gt.get("gasoline_change_phrase") else None,
                crude_avg_ok=eval_result["crude_avg_ok"] if gt.get("crude_avg_phrase") else None,
                gasoline_avg_ok=eval_result["gasoline_avg_ok"] if gt.get("gasoline_avg_phrase") else None,
                overall_accuracy=eval_result["overall_accuracy"],
            )
        )

        # 更新聚合计数
        # crude_change
        if gt.get("crude_change_phrase"):
            agg_denoms["crude_change_ok"] += 1
            if eval_result["crude_change_ok"]:
                agg_hits["crude_change_ok"] += 1
        # gasoline_change
        if gt.get("gasoline_change_phrase"):
            agg_denoms["gasoline_change_ok"] += 1
            if eval_result["gasoline_change_ok"]:
                agg_hits["gasoline_change_ok"] += 1
        # crude_avg
        if gt.get("crude_avg_phrase"):
            agg_denoms["crude_avg_ok"] += 1
            if eval_result["crude_avg_ok"]:
                agg_hits["crude_avg_ok"] += 1
        # gasoline_avg
        if gt.get("gasoline_avg_phrase"):
            agg_denoms["gasoline_avg_ok"] += 1
            if eval_result["gasoline_avg_ok"]:
                agg_hits["gasoline_avg_ok"] += 1

    # 3) 打印 summary 表格（逐周）
    print("\n================ SUMMARY PER WEEK ================")
    header = (
        f"{'Week':<12} | {'CrudeΔ':<7} | {'GasΔ':<7} | "
        f"{'Crude5yr':<9} | {'Gas5yr':<9} | {'Overall':<7}"
    )
    print(header)
    print("-" * len(header))
    
    def fmt(v):
        return "-" if v is None else str(v)

    for r in results_per_week:
        print(
            f"{r['week']:<12} | "
            f"{fmt(r['crude_change_ok']):<7} | "
            f"{fmt(r['gasoline_change_ok']):<7} | "
            f"{fmt(r['crude_avg_ok']):<9} | "
            f"{fmt(r['gasoline_avg_ok']):<9} | "
            f"{r['overall_accuracy']:<7.2f}"
        )

    # 4) 打印总体 accuracy
    print("\n================ OVERALL ACCURACY ================")
    overall_stats = {}
    for k in agg_hits:
        denom = agg_denoms[k]
        overall_stats[k] = agg_hits[k] / denom if denom else 0.0

    print(json.dumps(overall_stats, indent=2))
    print("==================================================\n")


if __name__ == "__main__":
    main()
