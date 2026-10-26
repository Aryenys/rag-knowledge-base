"""
MCP Server 入口：让 Claude Desktop / Cursor / Cline 等任意 MCP 客户端都能查我们的知识库。

【这个文件的定位】
任务1 做的是"自家 Agent 能调工具"（私有协议：OpenAI tool_calling，跑在 HTTP 服务里）。
任务2 做的是"别人的 Agent 也能调我们的工具"（开放协议：MCP，跑在独立 stdio 进程里）。
两者调用的是**同一个** app/agent/tools.py 里的 run_tool —— 业务逻辑只有一份，
区别只在于"谁在问、用什么格式问"。这是本文件最值得在面试里讲的一点。

【为什么 MCP 值得做（面试话术）】
没有 MCP 时，N 个 AI 应用要用我的知识库，就得写 N 套适配代码（每个应用的工具格式都不一样）。
有了 MCP，我只需要写**这一份** server，所有支持 MCP 的客户端都能直接连——
把 N×M 的适配问题（N 个应用 × M 个数据源）降成 N+M。这就是它被称为
"AI 应用的 USB-C 接口"的原因。

【两个必须处理的坑（都是 stdio 传输特有的）】
坑1：stdout 只能给协议用。MCP 的 stdio 传输下，客户端通过进程的**标准输出**读 JSON-RPC 报文。
     业务代码里任何一句 print 都会插进这条流里，导致客户端 JSON 解析失败、工具调用莫名报错。
     而我们的 embedder.py 里就有 print("embedding 第 x/y 批...")。
     解法：工具执行期间用 contextlib.redirect_stdout 把 print 引到 stderr，执行完立刻恢复。
坑2：.env 加载依赖当前工作目录。Claude Desktop 拉起这个进程时 cwd 是不确定的，
     app/config.py 里的 load_dotenv()（无参数）会找不到 .env，导致 API key 全空。
     解法：在 import app.* 之前，用显式路径 load_dotenv(PROJECT_ROOT / ".env")。
"""
import contextlib
import logging
import os
import sys
from pathlib import Path

# ---------- 先确定项目根，再显式加载 .env（顺序不能反，见上面坑2）----------
# Path(__file__).resolve() 拿到的是本文件的绝对路径，与客户端的 cwd 无关，
# 这样无论谁启动这个进程，都能找到同一份 .env。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=False)

# ---------- 日志一律走 stderr，绝不碰 stdout（见上面坑1）----------
# 注意这里不用 logging.basicConfig：basicConfig 会改 root logger，
# 把 httpx 等第三方库的 INFO 日志也一起打开，非常吵。
# 只给自己的 logger 挂 handler，就不影响别人。
logger = logging.getLogger("rag-mcp")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s [mcp] %(levelname)s %(message)s"))
logger.addHandler(_handler)

from mcp.server.mcpserver import MCPServer
from pydantic import Field
from typing import Annotated

# 复用任务1 已经写好的工具分发入口：检索链路一行都不用重写
from app.agent.tools import run_tool

# MCPServer 是 mcp 2.x 里 FastMCP 的新名字（1.x 叫 FastMCP，2.x 改名并调整了部分 API）
mcp = MCPServer(
    name="rag-knowledge-base",
    description="企业知识库检索：混合召回（BM25+向量） + RRF 融合 + BGE-Rerank 精排",
    version="0.2.0",
)


@mcp.tool(
    description=(
        "检索企业知识库的正文内容，返回相关文档片段。"
        "当用户的问题需要知识库里文档的具体内容才能回答时必须调用它，"
        "包括：某份文档讲了什么、某个技术概念是什么、某个流程/制度/参数是什么。"
    )
)
def search_knowledge_base(
    query: Annotated[
        str,
        Field(description="检索用的问题或关键词，必须独立完整，不要包含「它」「这个」之类的指代"),
    ],
) -> str:
    """执行检索并返回给模型看的参考资料文本。"""
    # redirect_stdout 只在工具执行期间生效，执行完立即恢复，
    # 因此不会影响 MCP SDK 自己往 stdout 写 JSON-RPC 报文。
    with contextlib.redirect_stdout(sys.stderr):
        result = run_tool("search_knowledge_base", {"query": query})

    logger.info("检索「%s」: %s", query, result["summary"])
    return result["for_model"]


@mcp.tool(
    description=(
        "获取知识库的元信息：文档总数、切片总数、文件清单。"
        "只在用户只想知道『知识库里有多少东西、装了哪些文件』这类纯数量/清单问题时使用。"
        "想知道某份文档讲了什么，必须用 search_knowledge_base。"
    )
)
def get_knowledge_base_stats() -> str:
    """返回知识库统计信息（只读 MySQL 元数据，不碰向量库，毫秒级返回）。"""
    with contextlib.redirect_stdout(sys.stderr):
        result = run_tool("get_knowledge_base_stats", {})

    logger.info("统计工具: %s", result["summary"])
    return result["for_model"]


def main() -> None:
    """启动 MCP Server。

    transport 说明：
      stdio          —— 默认，客户端直接拉起本进程、用管道通信（Claude Desktop / Cursor 都用这个）
      streamable-http —— 需要多客户端共享一个 server 时用，本项目暂不需要
    """
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    logger.info("MCP Server 启动，transport=%s，项目根=%s", transport, PROJECT_ROOT)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
