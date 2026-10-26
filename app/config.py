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
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))  # 相似度阈值，低于则视为无关召回

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

    # ---------- Agent（任务1） ----------
    # 工具调用最大轮数：防死循环硬闸门。达到上限后不再把 tools 传给模型，
    # 模型没有工具可调，只能基于已有信息给最终答案。
    AGENT_MAX_TOOL_ROUNDS: int = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "3"))

    # ---------- 缓存（任务3） ----------
    # 总开关：关掉后所有缓存逻辑直接短路，行为与加缓存前完全一致（可一键回滚）
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
    # 检索结果缓存：省掉 embedding + BM25 + Rerank 三次开销
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "300"))                # 单条缓存存活秒数
    CACHE_MAX_ENTRIES: int = int(os.getenv("CACHE_MAX_ENTRIES", "500"))  # 最多缓存多少条（超出按 LRU 淘汰）
    # 知识库指纹的本地缓存秒数：避免每次检索都查一次 MySQL
    KB_FINGERPRINT_TTL: int = int(os.getenv("KB_FINGERPRINT_TTL", "10"))
    # 答案缓存：省掉 LLM 生成。默认关闭——生成带随机性且要"假流式"回放，
    # 只有在"同一问题被反复问"的场景才划算，交给部署方决定。
    CACHE_ANSWER_ENABLED: bool = os.getenv("CACHE_ANSWER_ENABLED", "false").lower() in ("1", "true", "yes")


settings = Settings()
