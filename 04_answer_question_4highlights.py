import sys
import textwrap
import requests
import re
import streamlit as st
import json
import os

import snowflake.connector
from snowflake_config import SF_CONFIG

from openai import OpenAI  # ✅ 新增：OpenAI 官方 SDK

# 向量检索仍然用 Cortex 的 embedding 模型
EMBED_MODEL = "snowflake-arctic-embed-l-v2.0"

# 相似度阈值：低于这个值就认为“和 WPSR Highlights 不相关”
SIMILARITY_THRESHOLD = 0.25

# LLM 选择：
# - 默认使用云端 GPT（gpt-4o-mini）
# - 如果你在本地 export USE_LOCAL_LLM=1，则改用本地 Ollama Llama3
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "0") == "1"

# 默认 GPT 模型名（性价比高）
DEFAULT_GPT_MODEL = os.getenv("GPT_MODEL_NAME", "gpt-4o-mini")


# =========================================================
# 从 WPSR Highlights 样板中提取的关键词（再加一些通用词）
# 用于：硬过滤“问题是否大概率和 WPSR 有关”
# =========================================================
WPSR_KEYWORDS = [
    # 报告本身 &总类
    "wpsr",
    "weekly petroleum status report",
    "petroleum",
    "crude",
    "crude oil",
    "u.s. crude oil",
    "west texas intermediate",
    "wti",

    # 炼厂 / 加工
    "refinery",
    "refineries",
    "refinery inputs",
    "refinery input",
    "refinery utilization",
    "operable capacity",

    # 产量 / 供应
    "production",
    "gasoline production",
    "distillate fuel production",
    "total products supplied",
    "product supplied",
    "motor gasoline product supplied",
    "distillate fuel product supplied",
    "jet fuel product supplied",

    # 进出口
    "import",
    "imports",
    "crude oil imports",
    "gasoline imports",
    "motor gasoline imports",
    "distillate fuel imports",
    "export",
    "exports",

    # 库存 / 存量
    "inventory",
    "inventories",
    "stocks",
    "crude oil inventories",
    "commercial crude oil inventories",
    "commercial petroleum inventories",
    "total commercial petroleum inventories",
    "motor gasoline inventories",
    "distillate fuel inventories",
    "propane",
    "propylene",
    "propane/propylene inventories",
    "strategic petroleum reserve",
    "spr",
    "five year average",
    "five-year average",

    # 品种
    "gasoline",
    "motor gasoline",
    "finished gasoline",
    "gasoline blending components",
    "distillate",
    "distillate fuel",
    "jet fuel",
    "heating oil",
    "no. 2 heating oil",
    "diesel",
    "diesel fuel",
    "retail diesel fuel",
    "regular gasoline",

    # 价格相关
    "price",
    "prices",
    "spot price",
    "retail price",
    "national average retail price",
    "per gallon",
    "per barrel",
    "new york harbor",
    "new york harbor spot price",

    # 单位 / 时间表达
    "million barrels per day",
    "thousand barrels per day",
    "million barrels",
    "barrels per day",
    "week ending",
    "last week",
    "past four weeks",
    "same period last year",
]

# ---------- Snowflake 连接 ----------

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


# ---------- 简单的意图分类（基于关键词） ----------

def classify_question(question: str) -> str:
    q = question.lower()

    if "latest" in q or "current" in q or "most recent" in q:
        return "latest"

    if "trend" in q or "past few weeks" in q or "over the past" in q:
        return "trend"

    if "price" in q or "retail" in q or "wti" in q:
        return "price"

    if "gasoline" in q:
        return "gasoline"

    if "distillate" in q:
        return "distillate"

    if "import" in q:
        return "imports"

    return "general"


# ---------- 判断问题是否大概率和 WPSR 有关（硬过滤） ----------

def is_potentially_wpsr_related(question: str) -> bool:
    """
    如果问题里完全不包含任何 WPSR 相关关键词，就认为不该走 WPSR RAG。
    """
    q = question.lower()
    return any(kw in q for kw in WPSR_KEYWORDS)


# ---------- 在 Snowflake 里检索 Top-K 相关 Highlights chunks ----------

def retrieve_top_chunks(conn, question: str, top_k: int = 6, weeks_back: int = 12):
    """
    根据问题意图动态调整检索策略：
      - latest: 强推最近几周、top_k 较小
      - trend: weeks_back 可以稍大，top_k 可以适当放大
      - price/imports: 改用 chunk_text 做关键词过滤（因为 topics 里没有这类标签）
      - gasoline/distillate: 使用 topics + text 双重过滤，减少噪音
      - 如果问题里出现 "week ending YYYY-MM-DD"，则强制只用该周的数据
    """

    intent = classify_question(question)

    # --- 从问题里解析 "week ending YYYY-MM-DD" 作为 week_hint ---
    week_hint = None
    m = re.search(r"week ending\s+(\d{4}-\d{2}-\d{2})", question.lower())
    if m:
        week_hint = m.group(1)

    # 1) 动态控制时间窗口和 top_k（在用户传入的上限之内调整）
    if intent == "latest":
        effective_weeks_back = min(weeks_back, 4)
        effective_top_k = min(top_k, 4)
    elif intent == "trend":
        effective_weeks_back = min(weeks_back, 8)  # 趋势问题可以放宽一点
        effective_top_k = min(top_k * 2, 8)
    else:
        effective_weeks_back = weeks_back
        effective_top_k = top_k

    # 2) topic / text 过滤条件
    topic_filter = ""
    if intent == "gasoline":
        # topics 里有 gasoline，同时用正文兜底
        topic_filter = "AND (gh.topics ILIKE '%%gasoline%%' OR gh.chunk_text ILIKE '%%gasoline%%')"
    elif intent == "distillate":
        topic_filter = "AND (gh.topics ILIKE '%%distillate%%' OR gh.chunk_text ILIKE '%%distillate%%')"
    elif intent == "imports":
        # topics 里没有 imports，只能在正文里找
        topic_filter = "AND gh.chunk_text ILIKE '%%import%%'"
    elif intent == "price":
        # price 相关只能从正文里搜关键词
        topic_filter = (
            "AND (gh.chunk_text ILIKE '%%price%%' "
            "OR gh.chunk_text ILIKE '%%wti%%' "
            "OR gh.chunk_text ILIKE '%%brent%%' "
            "OR gh.chunk_text ILIKE '%%retail%%')"
        )

    # 3) 排序策略：latest 强调“最近 + 相似度”，其他则按相似度优先
    order_clause = (
        "ORDER BY week_ending DESC, score DESC"
        if intent == "latest"
        else "ORDER BY score DESC"
    )

    # --- 如果识别到具体 week_ending，则强制过滤该周 ---
    week_clause = ""
    if week_hint:
        week_clause = f"AND gh.week_ending = TO_DATE('{week_hint}')"

    sql = f"""
    WITH query AS (
      SELECT AI_EMBED('{EMBED_MODEL}', %s) AS q_vec
    ),
    scored AS (
      SELECT
        gh.chunk_id,
        gh.document_id,
        gh.week_ending,
        gh.topics,
        gh.chunk_text,
        VECTOR_COSINE_SIMILARITY(gh.embedding, query.q_vec) AS score
      FROM RAW.GOLD_CHUNK_EMBEDDING_HIGHLIGHTS gh,
           query
      WHERE gh.week_ending >= DATEADD(week, -{effective_weeks_back}, CURRENT_DATE())
        AND gh.week_ending IS NOT NULL
        AND gh.chunk_text NOT ILIKE '%%Figure %%'
        AND gh.chunk_text NOT ILIKE '%%Table %%'
        {topic_filter}
        {week_clause}
    )
    SELECT
      chunk_id,
      document_id,
      week_ending,
      topics,
      score,
      chunk_text
    FROM scored
    {order_clause}
    LIMIT {effective_top_k};
    """

    cur = conn.cursor()
    try:
        cur.execute(sql, (question,))
        rows = cur.fetchall()
        cols = [c[0].lower() for c in cur.description]
    finally:
        cur.close()

    results = [dict(zip(cols, r)) for r in rows]

    # 简单 debug：方便以后排查检索行为
    if not results:
        print(
            f"[DEBUG] intent={intent}, week_hint={week_hint}, "
            f"weeks_back={effective_weeks_back}, top_k={effective_top_k}, "
            f"no rows returned"
        )
        return []

    top_score = results[0]["score"]
    print(
        f"[DEBUG] intent={intent}, week_hint={week_hint}, "
        f"weeks_back={effective_weeks_back}, top_k={effective_top_k}, "
        f"top_score={top_score:.4f}"
    )

    # 相似度阈值过滤：如果最高分太低，认为“没有可靠的 WPSR 上下文”
    if top_score < SIMILARITY_THRESHOLD:
        print(
            f"[DEBUG] top_score {top_score:.4f} < threshold {SIMILARITY_THRESHOLD:.2f} → treat as no relevant context"
        )
        return []

    return results


# ---------- 构造给 LLM 的 prompt ----------

def build_prompt(question: str, chunks):
    """
    强约束 Highlights Prompt：
      - 先 SUPPORTING_SENTENCES（原句逐字抄，不许改数字）
      - 再 ANSWER（总结，但所有数字必须来自 SUPPORTING_SENTENCES）
      - 明确要求必须说明 five-year average 对比情况（如果上下文有）
      - 禁止错误“猜测式解释”或漏掉问题中提到的品种
    """
    # 没有任何 chunks（相似度过低或检索不到），直接让模型说 "I don't know"
    if not chunks:
        return f"""You are an energy analyst.
You have no relevant WPSR Highlights context for this question and must answer exactly:
"I could not find relevant information in the provided WPSR Highlights database to answer your question."

Question: {question}
"""

    # 拼 CONTEXT
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] Week ending: {c['week_ending']} | Document: {c['document_id']} | Topics: {c['topics']}"
        body = c["chunk_text"]
        block = header + "\n" + body
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    # 把最硬的几条规则集中放在前面，方便小模型抓住
    system_rules = """
You are an expert crude & products market analyst at an energy trading firm.
You answer questions ONLY using the information in the CONTEXT below, which comes from the Highlights section
of the EIA Weekly Petroleum Status Report (WPSR).

HARD RULES (MUST OBEY):
1. If the QUESTION is not clearly related to WPSR topics (inventories, refinery utilization, production, imports,
   exports, product supplied, prices, retail prices, crude oil, gasoline, distillate, jet fuel, diesel, propane),
   you MUST answer exactly:
   "I could not find relevant information in the provided WPSR Highlights database to answer your question."
2. You MUST NOT use any knowledge outside the CONTEXT.
3. Every numeric value in your answer MUST appear verbatim in the CONTEXT. No rounding or estimation.
4. If the QUESTION mentions a specific product or topic (crude oil, gasoline, distillate, jet fuel, diesel, imports,
   inventories, prices, etc.), your ANSWER MUST explicitly mention that product or topic by name.
"""

    prompt = f"""
{system_rules}

Your tone should be:
- professional and confident
- natural and fluent, like a market analyst briefing colleagues
- not robotic
- you may paraphrase, but do NOT distort meaning or numbers

==================== RULES ====================

STRICT RELEVANCE RULE (VERY IMPORTANT):
- You MUST only answer if the QUESTION is clearly related to the WPSR Highlights content,
  including but not limited to:
  inventories, refinery utilization, production, imports, exports,
  stock levels, product supplied, prices, retail prices.
- If the QUESTION is unrelated, respond:
  "I could not find relevant information in the provided WPSR Highlights database to answer your question."

KEYWORD COVERAGE RULE (VERY IMPORTANT):
- If the QUESTION explicitly mentions a product or topic (e.g., “distillate fuel”, “gasoline”, “crude oil”, “imports”, “prices”),
  then your answer MUST explicitly talk about that topic using its name.
- Do NOT omit the keyword from your answer.
- If the keyword appears multiple times in CONTEXT, you MUST cover ALL relevant pieces of information about it,
  not just one.

COMPREHENSIVENESS RULE:
- If multiple distinct types of information exist for the topic (e.g., production, inventories, imports, stocks, retail prices),
  summarize ALL of them.
- Do NOT selectively mention one and ignore the others if they exist in CONTEXT.

NUMERIC RULES:
- Every numeric value in your answer MUST appear verbatim in the CONTEXT.
- No approximation, no estimation, no rounding, no calculation.

CITATION RULES:
- Each CONTEXT block is indexed in order: [1], [2], [3], …
- Each block begins with "Week ending: <DATE>".
- When you cite supporting evidence, you MUST indicate which block it comes from AND the week-ending date.

==================== PROCEDURE ====================

Step 0 — RELEVANCE CHECK:
- If NOT related to WPSR Highlights → reply with the polite unavailable message and STOP.

Step 1 — SUPPORTING_SENTENCES:
- Collect ALL sentences or statements in CONTEXT that are relevant to the topic mentioned in the QUESTION.
- You may paraphrase lightly, but you MUST keep all numeric values exactly as in CONTEXT.
- For EACH bullet, you MUST append the source block index and the exact week-ending date from that block.

Required format for EACH bullet:
- "<brief paraphrased supporting statement>" (source: [block_index], week ending <DATE>)

Example:
- "U.S. commercial crude oil inventories increased by 0.4 million barrels last week and remain about 3% below the five-year average." (source: [1], week ending December 19, 2025)

Step 2 — ANSWER LOGIC:

CASE A — Exact or clearly relevant information EXISTS:
- Provide a clear, natural, analyst-style 2–4 sentence summary.
- MUST explicitly mention the keyword topic.
- MUST cover ALL relevant aspects present in the CONTEXT
  (e.g., production, inventories, imports, retail prices, etc., if they exist).
- Natural tone. No robotic repetition.

CASE B — No relevant information at all:
- Say:
  "I don't know based on the provided WPSR excerpts."

==================== OUTPUT FORMAT ====================

ANSWER:
<analyst-style response following the logic above>

SUPPORTING_SENTENCES:
- "<statement 1>" (source: [block_index], week ending <DATE>)
- "<statement 2>" (source: [block_index], week ending <DATE>)
- "..."

QUESTION:
{question}

CONTEXT:
{context_text}

Now produce the ANSWER and SUPPORTING_SENTENCES following the rules and format above.
"""

    return textwrap.dedent(prompt).strip()


# ---------- GPT 云端 LLM ----------

def _get_openai_client() -> OpenAI:
    """
    从环境变量或 Streamlit secrets 中获取 OPENAI_API_KEY，
    并返回一个 OpenAI 客户端。如果没有配置，抛出友好错误。
    """
    api_key = os.getenv("OPENAI_API_KEY")

    # 如果环境变量没有，尝试从 st.secrets 读取（本地/.streamlit 或云端 Secrets）
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please configure it either as an "
            "environment variable or in Streamlit secrets."
        )

    return OpenAI(api_key=api_key)


def call_gpt_llm(prompt: str, model: str = None) -> str:
    """
    调用云端 GPT 模型（默认 gpt-4o-mini），用于 Streamlit Cloud 等环境。
    """
    if model is None:
        model = DEFAULT_GPT_MODEL

    client = _get_openai_client()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an energy market analyst specialized in the "
                    "EIA Weekly Petroleum Status Report (WPSR) Highlights."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ---------- 本地 Ollama LLM（可选） ----------

def call_local_llm(prompt: str) -> str:
    """
    调用本地 Ollama Llama3.2 (3B) 模型。
    只有在设置 USE_LOCAL_LLM=1 时才会被使用。
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": True,
        "num_predict": 200,
        "temperature": 0.1,  # 稍微调低，让模型更听话
    }

    res = requests.post(url, json=payload, stream=True, timeout=120)
    res.raise_for_status()

    full_text = ""
    for line in res.iter_lines():
        if not line:
            continue
        data = json.loads(line.decode("utf-8"))
        if data.get("done"):
            break
        full_text += data.get("response", "")

    return full_text


# ---------- 主流程：检索 + 生成 ----------

def answer_question(question: str, top_k: int = 4, weeks_back: int = 8):
    # 第一层硬过滤：如果问题里完全没有 WPSR 相关关键词，直接不给 WPSR 答复
    if not is_potentially_wpsr_related(question):
        msg = "I could not find relevant information in the provided WPSR Highlights database to answer your question."
        print(f"[INFO] Question seems unrelated to WPSR. Returning fallback message.\n{msg}")
        return msg

    conn = sf_connect()
    try:
        # 1) 先检索 Top-K 相关 Highlights chunks（内部会根据 intent / week_hint 调整）
        chunks = retrieve_top_chunks(conn, question, top_k=top_k, weeks_back=weeks_back)

        print(f"[INFO] Retrieved {len(chunks)} highlight chunks for question.")
        for i, c in enumerate(chunks, start=1):
            prev = c["chunk_text"][:120].replace("\n", " ")
            print(f"  [{i}] week={c['week_ending']}  score={c.get('score', 0.0):.4f}  topics={c['topics']}")
            print(f"      preview: {prev}")
        print()
    finally:
        conn.close()

    # 2) 构建 prompt（如果 chunks 为空，会自动走 "I don't know" 模式）
    prompt = build_prompt(question, chunks)

    # 3) 生成回答：优先用云端 GPT；如果你设置了 USE_LOCAL_LLM=1，则用本地 Llama
    try:
        if USE_LOCAL_LLM:
            print("[INFO] Using local Llama3 (Ollama) as LLM backend.")
            answer = call_local_llm(prompt)
        else:
            print(f"[INFO] Using cloud GPT model '{DEFAULT_GPT_MODEL}' as LLM backend.")
            answer = call_gpt_llm(prompt)
    except RuntimeError as e:
        # OPENAI_API_KEY 未配置等错误
        err_msg = f"LLM configuration error: {e}"
        print(f"[ERROR] {err_msg}")
        return err_msg
    except Exception as e:
        print(f"[ERROR] Unexpected error when calling LLM: {e}")
        return f"Error while answering: {e}"

    print("\n================= MODEL ANSWER =================\n")
    print(answer)
    print("\n=============================================================\n")

    return answer


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Please enter your question:\n> ")

    answer_question(question, top_k=6, weeks_back=12)


if __name__ == "__main__":
    main()
