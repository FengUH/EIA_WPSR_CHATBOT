import importlib.util
from pathlib import Path
import time
import html  # 备用，虽然现在主要用 markdown 直接展示
import streamlit as st

# ================== 动态加载 04_answer_question_4highlights.py ==================

SCRIPT_DIR = Path(__file__).resolve().parent
ANSWER_SCRIPT = SCRIPT_DIR / "04_answer_question_4highlights.py"

spec = importlib.util.spec_from_file_location("answer_hl", ANSWER_SCRIPT)
answer_hl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answer_hl)


# ================== 复用后端逻辑 ==================

def _call_llm_backend(prompt: str) -> str:
    """
    统一的 LLM 调用入口：
      - 如果 04_answer_question_4highlights.py 里设置了 USE_LOCAL_LLM=True，
        则调用本地 Ollama (call_local_llm)
      - 否则调用云端 GPT (call_gpt_llm)
    这样只改这一处，就能保持原 app.py 的所有表现方式不变。
    """
    use_local = getattr(answer_hl, "USE_LOCAL_LLM", False)
    if use_local:
        print("[INFO] Using local Llama backend from app.py")
        return answer_hl.call_local_llm(prompt)
    else:
        print("[INFO] Using cloud GPT backend from app.py")
        return answer_hl.call_gpt_llm(prompt)


def answer_question_ui(question: str, top_k: int = 6, weeks_back: int = 12) -> str:
    conn = answer_hl.sf_connect()
    try:
        chunks = answer_hl.retrieve_top_chunks(
            conn,
            question,
            top_k=top_k,
            weeks_back=weeks_back,
        )
    finally:
        conn.close()

    prompt = answer_hl.build_prompt(question, chunks)
    # ⭐ 这里改成调用统一入口（内部再决定本地 / GPT）
    answer = _call_llm_backend(prompt)
    return answer


def answer_question_ui_with_timing(
    question: str,
    top_k: int = 4,
    weeks_back: int = 8,
):
    t0 = time.time()

    conn = answer_hl.sf_connect()
    try:
        t1 = time.time()
        chunks = answer_hl.retrieve_top_chunks(
            conn,
            question,
            top_k=top_k,
            weeks_back=weeks_back,
        )
        t2 = time.time()
    finally:
        conn.close()

    prompt = answer_hl.build_prompt(question, chunks)
    t3 = time.time()

    # ⭐ 这里也改成统一入口（保留原 timing 结构）
    answer = _call_llm_backend(prompt)
    t4 = time.time()

    timings = {
        "retrieval_seconds": round(t2 - t1, 3),
        "prompt_build_seconds": round(t3 - t2, 3),
        "llm_seconds": round(t4 - t3, 3),
        "total_seconds": round(t4 - t0, 3),
    }

    print(
        "[TIMING] Retrieval: "
        f"{timings['retrieval_seconds']}s, "
        f"Prompt build: {timings['prompt_build_seconds']}s, "
        f"LLM: {timings['llm_seconds']}s, "
        f"TOTAL: {timings['total_seconds']}s"
    )

    return answer, timings


# ================== 将主回答与 SUPPORTING_SENTENCES 分离 ==================

def split_answer(answer: str):
    """
    把回答切成两部分：
      - main_answer: SUPPORTING_SENTENCES 之前
      - extra_answer: 从 SUPPORTING_SENTENCES 起直到结尾
    """
    lower = answer.lower()
    keys = [
        "supporting_sentences:",
        "supporting_sentences",
        "supporting sentences:",
        "supporting sentences",
    ]

    idx = -1
    for k in keys:
        idx = lower.find(k)
        if idx != -1:
            break

    if idx == -1:
        return answer.strip(), ""

    main = answer[:idx].rstrip()
    extra = answer[idx:].lstrip()
    return main, extra


# ================== ChatWPSR UI ==================

def main():
    st.set_page_config(
        page_title="ChatWPSR",
        page_icon="🗂️",
        layout="centered",
    )

    # ---- 样式 ----
    st.markdown(
        """
        <style>
        .stApp { background-color: #ffffff; }

        .main .block-container {
            max-width: 840px;
            padding-top: 5rem;
            padding-bottom: 5rem;
            margin: 0 auto;
        }

        .chat-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            color: #0f172a;
        }

        .chat-subtitle {
            text-align: center;
            font-size: 0.95rem;
            color: #6b7280;
            margin-bottom: 2.6rem;
        }

        .user-bubble {
            background: #f3f4f6;
            padding: 12px 14px;
            border-radius: 16px;
            max-width: 75%;
            float: right;
            margin-top: 22px;
            margin-bottom: 26px;
        }

        .assistant-bubble {
            background: transparent;
            padding: 6px 2px;
            max-width: 85%;
            float: left;
            margin-top: 10px;
            margin-bottom: 26px;
        }

        .clearfix { clear: both; }

        /* expander 外观 */
        details {
            border-radius: 14px;
            border: 1px solid #e5e7eb;
            background: linear-gradient(180deg, #f9fafb 0%, #f3f4f6 100%);
            padding: 0;
            margin-top: 0.7rem;
            margin-bottom: 1.2rem;
            transition: all 0.18s ease-in-out;
        }

        details:hover {
            border-color: #d1d5db;
            background: linear-gradient(180deg, #ffffff 0%, #f5f6f8 100%);
            box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        }

        details[open] {
            box-shadow: 0 10px 28px rgba(0,0,0,0.06);
        }

        details > summary {
            list-style: none;
            cursor: pointer;
            padding: 0.55rem 1.0rem;
            font-size: 0.9rem;
            font-weight: 600;
            color: #374151;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        details > div {
            padding: 0.5rem 1.0rem 0.9rem 1.0rem;
        }

        /* 默认问题按钮 pill 样式 */
        div.stButton > button {
            width: 100%;
            border-radius: 999px;
            border: 1px solid #e5e7eb;
            background-color: #f9fafb;
            color: #4b5563;
            font-size: 0.82rem;
            padding: 0.45rem 0.8rem;
            text-align: left;
            white-space: normal;
            height: auto;
        }

        div.stButton > button:hover {
            border-color: #c7d2fe;
            background-color: #eef2ff;
            color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- 会话状态 ----
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_timings" not in st.session_state:
        st.session_state.last_timings = None
    if "last_extra" not in st.session_state:
        st.session_state.last_extra = ""

    # ---- Header ----
    st.markdown(
        """
        <div class="chat-title">ChatWPSR</div>
        <div class="chat-subtitle">
          Query the Weekly Petroleum Status Report.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- 顶部 Guided questions（只出现一次） ----
    SUGGESTED_QUESTIONS = [
        "How did crude inventories change vs last week in the latest report?",
        "How did national gasoline and diesel retail prices change in the latest report?",
        "What happened to the WTI price vs last week and a year ago in the latest report?",
        "How did gasoline and distillate stocks move vs the 5-year average in the latest report?",
    ]

    chosen_question = None
    with st.expander("Guided questions to explore the WPSR", expanded=False):
        st.markdown(
            '<div style="font-size:0.8rem; color:#6b7280; margin-bottom:0.35rem;">'
            'Pick a starter question or type your own below.'
            '</div>',
            unsafe_allow_html=True,
        )
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if st.button(q, key=f"suggest_q_{i}"):
                chosen_question = q

    # ---- 聊天历史 ----
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div><div class="clearfix"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">{msg["content"]}</div><div class="clearfix"></div>',
                unsafe_allow_html=True,
            )

    # ---- 输入框 ----
    user_input = st.chat_input("Ask a question about WPSR…")

    # ---- 统一入口 ----
    question = chosen_question or (user_input.strip() if user_input else None)

    if question:
        # 用户气泡
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(
            f'<div class="user-bubble">{question}</div><div class="clearfix"></div>',
            unsafe_allow_html=True,
        )

        # LLM + 计时
        with st.spinner("Thinking with WPSR Highlights..."):
            try:
                raw_answer, timings = answer_question_ui_with_timing(
                    question,
                    top_k=4,
                    weeks_back=8,
                )
            except Exception as e:
                raw_answer = f"Error while answering: {e}"
                timings = None

        main_answer, extra_answer = split_answer(raw_answer)

        # 助手主回答气泡
        st.session_state.messages.append({"role": "assistant", "content": main_answer})
        st.markdown(
            f'<div class="assistant-bubble">{main_answer}</div><div class="clearfix"></div>',
            unsafe_allow_html=True,
        )

        # ⭐ 这一轮的 Additional details：紧贴在回答后面
        if timings is not None or extra_answer:
            with st.expander("For Developer", expanded=False):
                if timings is not None:
                    st.subheader("Timing breakdown")
                    st.json(timings)
                if extra_answer:
                    st.subheader("Supporting sentences & extra context")
                    st.markdown(extra_answer)

        # 如你之后还想用到历史调试信息，再存一份（虽然当前 UI 不用它们）
        st.session_state.last_timings = timings
        st.session_state.last_extra = extra_answer


if __name__ == "__main__":
    main()
