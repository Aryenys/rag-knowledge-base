"""
Agent 能力包（任务1：Tool Calling + Agent 决策层）。

- tools.py         工具层：知识库检索 / 知识库统计，附 OpenAI tools 的 JSON Schema
- agent_service.py 决策层：模型自主决定"直接回答 / 调用工具"，支持多轮工具调用

与普通模式（app.services.chat_service）完全并行，互不影响：
普通模式走"写死的检索->生成"链路，Agent 模式把"要不要检索"交给模型判断。
"""
