import snowflake.connector
from snowflake_config import SF_CONFIG


# 加载 embedding 模型（1024 维）
EMBED_MODEL = "snowflake-arctic-embed-l-v2.0"

# Silver / Gold 表名
SILVER_TABLE_NAME = "SILVER_CHUNK_HIGHLIGHTS"
GOLD_TABLE_NAME   = "GOLD_CHUNK_EMBEDDING_HIGHLIGHTS"


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


def ensure_gold_table(conn):
    """
    创建 GOLD_CHUNK_EMBEDDING_HIGHLIGHTS 表
    embedding 列是 VECTOR(FLOAT, 1024)，匹配 arctic-embed-l-v2.0。
    """
    sql = f"""
    CREATE TABLE IF NOT EXISTS {GOLD_TABLE_NAME} (
      chunk_id        STRING,
      document_id     STRING,
      chunk_index     INT,
      week_ending     DATE,
      topics          STRING,
      chunk_text      STRING,
      embedding       VECTOR(FLOAT, 1024),
      embedding_model STRING,
      created_at      TIMESTAMP_NTZ
    );
    """
    cur = conn.cursor()
    try:
        cur.execute(sql)
        print(f"[INIT] Ensured table {GOLD_TABLE_NAME} exists.")
    finally:
        cur.close()


def build_embeddings(conn):
    """
    调用 Snowflake Cortex 的 AI_EMBED 在 SQL 里生成向量，
    只给还没有 embedding 的 Highlights chunks 生成。
    """
    sql = f"""
    INSERT INTO {GOLD_TABLE_NAME} (
      chunk_id,
      document_id,
      chunk_index,
      week_ending,
      topics,
      chunk_text,
      embedding,
      embedding_model,
      created_at
    )
    SELECT
      sc.chunk_id,
      sc.document_id,
      sc.chunk_index,
      sc.week_ending,
      sc.topics,
      sc.chunk_text,
      AI_EMBED('{EMBED_MODEL}', sc.chunk_text),
      '{EMBED_MODEL}',
      CURRENT_TIMESTAMP()
    FROM RAW.{SILVER_TABLE_NAME} sc
    LEFT JOIN RAW.{GOLD_TABLE_NAME} ge
      ON sc.chunk_id = ge.chunk_id
    WHERE ge.chunk_id IS NULL;
    """
    cur = conn.cursor()
    try:
        cur.execute(sql)
        print(f"[INFO] Inserted {cur.rowcount} new embeddings into {GOLD_TABLE_NAME}.")
    finally:
        cur.close()


def main():
    conn = sf_connect()
    try:
        ensure_gold_table(conn)
        build_embeddings(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
