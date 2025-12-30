# snowflake_config.py
import os
import streamlit as st

def _get(key: str, default=None):
    # 优先从 Streamlit secrets 获取
    if "SNOWFLAKE" in st.secrets:
        return st.secrets["SNOWFLAKE"].get(key, default)
    # 其次从环境变量获取
    env_key = f"SNOWFLAKE_{key.upper()}"
    return os.getenv(env_key, default)

SF_CONFIG = {
    "user": _get("user"),
    "password": _get("password"),
    "account": _get("account"),
    "warehouse": _get("warehouse"),
    "database": _get("database"),
    "schema": _get("schema"),
    "role": _get("role"),
}
