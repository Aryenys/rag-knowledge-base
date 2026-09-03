"""
全局配置：所有环境变量集中在这里读取，其他模块一律 from app.config import settings
不要在代码里散落 os.getenv()，方便统一管理和面试时讲"配置中心"的设计。
"""
import os
from dotenv import load_dotenv

load_dotenv()  # 读取项目根目录的 .env 文件


class Settings:
    # ---------- LLM 生成（DeepSeek，兼容 OpenAI SDK） ----------
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")

    # ---------- Embedding / Rerank（硅基流动，有免费额度） ----------
    SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_BASE_URL: str = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    # ---------- 切分参数（W5 做对比实验时改这里） ----------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ---------- 检索参数 ----------
    VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "20"))   # 向量召回数量
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "20"))       # BM25 召回数量
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))    # rerank 后保留数量

    # ---------- 数据库 ----------
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://root:password@localhost:3306/rag_kb?charset=utf8mb4",
    )

    # ---------- 存储路径 ----------
    DATA_DIR: str = os.getenv("DATA_DIR", "data")                    # 上传文件目录
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "data/chroma")         # Chroma 持久化目录

    # ---------- 对话记忆 ----------
    HISTORY_TURNS: int = int(os.getenv("HISTORY_TURNS", "5"))        # 携带最近 N 轮历史


settings = Settings()
