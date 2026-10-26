"""
【任务2】MCP Server 冒烟测试：用真实的 MCP 客户端连真实的 server，验证协议层通。

为什么要单独写这个测试：
  MCP server 跑在独立 stdio 进程里，出问题时的现象是"客户端说工具调用失败"，
  但失败原因可能藏在子进程里（比如 stdout 被业务 print 污染、.env 没读到）。
  用真实客户端跑一遍，才能确认协议层是通的——这是 HTTP 接口测试覆盖不到的地方。

用法：
  python -m tests.test_mcp_server
预期：
  打印 2 个工具的 schema，然后打印两次真实调用的结果，最后输出"冒烟测试通过"。
"""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def main() -> None:
    # 用当前解释器启动 server 子进程，cwd 指向项目根（模拟客户端拉起进程）
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 握手：客户端声明协议版本与能力，server 返回 serverInfo
            init = await session.initialize()
            print(f"已连接 server: {init.server_info.name} v{init.server_info.version}")

            # ---------- 1. 列出工具 ----------
            tools_result = await session.list_tools()
            print(f"\n暴露了 {len(tools_result.tools)} 个工具：")
            for t in tools_result.tools:
                print(f"  - {t.name}")
                props = t.input_schema.get("properties", {})
                for pname, pinfo in props.items():
                    print(f"      参数 {pname}: {pinfo.get('description', '')[:40]}")
                if not props:
                    print("      （无参数）")

            if not tools_result.tools:
                raise SystemExit("没暴露任何工具，检查 @mcp.tool 装饰器是否生效")

            # ---------- 2. 真实调用统计工具 ----------
            print("\n--- 调用 get_knowledge_base_stats ---")
            stats = await session.call_tool("get_knowledge_base_stats", {})
            for c in stats.content:
                print(" ", c.text[:200])

            # ---------- 3. 真实调用检索工具（会触发 embedding + rerank API）----------
            print("\n--- 调用 search_knowledge_base ---")
            search = await session.call_tool(
                "search_knowledge_base", {"query": "RAG 相比纯生成模型有什么优势"}
            )
            for c in search.content:
                print(" ", c.text[:300])

            if not search.content:
                raise SystemExit("检索工具返回空，检查 run_tool 是否正常")

    print("\n冒烟测试通过：协议握手、工具发现、工具调用全部正常")


if __name__ == "__main__":
    asyncio.run(main())
