"""
【W6·Step1】日志配置：控制台 + 文件（按大小轮转，防止日志把磁盘撑爆）。

为什么需要它？
  之前排障时遇到 500，只能靠 uvicorn 窗口滚屏找 traceback —— 信息一闪而过，
  还会被后续请求冲掉。有了日志文件，错误现场能完整留存下来慢慢查。

轮转策略：单个文件 5MB，超过就改名成 app.log.1，最多留 5 个历史文件。
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
MAX_BYTES = 5 * 1024 * 1024   # 单文件上限 5MB
BACKUP_COUNT = 5              # 保留 5 个历史文件

FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

# 这几个第三方库正常运行时会打大量 DEBUG/INFO 日志，淹没我们自己的信息，统一调高门槛。
# 注意 httpx 的 logger 真名是 "httpx2"（不是 httpx），jieba 每次启动都会打 DEBUG 分词词典加载日志。
NOISY_LOGGERS = (
    "httpx", "httpx2", "httpcore", "chromadb", "openai",
    "urllib3", "jieba", "asyncio", "multipart",
)
_LEVEL = logging.INFO


def mute_noisy() -> None:
    """给第三方库降噪，并把根 logger 级别拉回来。

    为什么要拆成独立函数、分两次调用？
      因为很多库是"用到时才导入"的（比如 jieba 在 bm25_store 导入时才加载），
      而 setup_logging() 在 main.py 最顶部就跑了 —— 那时这些 logger 还没创建，
      设了也没用。所以应用启动完成后（依赖都导入齐了）要再调一次。
      另外有些库会自己调 logging.basicConfig() 改掉根级别，这里一并拉回来。
    """
    logging.getLogger().setLevel(_LEVEL)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def setup_logging(level: int = logging.INFO) -> None:
    """配置根 logger。做了幂等保护：uvicorn --reload 会重复执行模块，避免重复挂 handler。"""
    global _LEVEL
    _LEVEL = level

    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)
    formatter = logging.Formatter(FMT, DATEFMT)

    # ① 控制台：开发时直接看
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ② 文件：出事后翻旧账用
    LOG_DIR.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # ③ 给第三方库降噪（应用启动后会再调一次，覆盖那时才导入的库）
    mute_noisy()


def get_logger(name: str) -> logging.Logger:
    """各模块通过它拿 logger，名字用 __name__ 就能看出日志来自哪个文件"""
    return logging.getLogger(name)
