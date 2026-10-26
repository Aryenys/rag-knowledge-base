"""MCP Server 包：把知识库检索能力以 MCP 协议暴露给外部客户端。

与 app/agent（自家 Agent 用的工具层）的关系：
    同一套业务逻辑（app/agent/tools.py 的 run_tool），两种协议外壳。
    - 自家前端：HTTP + SSE，走 OpenAI tool_calling 协议
    - 外部客户端（Claude Desktop / Cursor）：stdio + JSON-RPC，走 MCP 协议
"""
