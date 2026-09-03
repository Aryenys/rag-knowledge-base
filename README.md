# RAG Knowledge Base —— 知识库智能问答系统

基于 RAG（检索增强生成）的企业知识库问答系统。支持上传 PDF / Markdown / TXT / DOCX 文档，  
自动切分、向量化入库，通过 **BM25 + 向量混合召回 + BGE-Rerank 重排序** 检索相关内容，  
调用 DeepSeek 生成带引用来源的回答，SSE 流式输出，支持多轮对话。

## 技术栈

Python · FastAPI · LangChain(text-splitters) · Chroma · MySQL · DeepSeek API · 硅基流动 API(BGE-M3 / BGE-Reranker) · Docker

## 架构图

```
前端(AI生成) ──HTTP/SSE──> FastAPI ──> RAG Pipeline ──> Chroma(向量) / BM25(关键词)
                              │                │
                          MySQL(元数据)    DeepSeek(生成) / 硅基流动(Embedding+Rerank)
```

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. 配置 API Key
copy .env.example .env        # 然后编辑 .env 填入 DeepSeek / 硅基流动的 Key

# 3. 启动
uvicorn app.main:app --reload --port 8000
# 打开 http://localhost:8000/docs 查看接口文档
```

## 开发进度

- [ ] W1：离线入库链路（loader / splitter / embedder / vector_store）
- [ ] W2：问答 API + SSE 流式 + 前端 MVP
- [ ] W3：混合检索 + Rerank + 查询改写
- [ ] W4：MySQL 持久化 + 多轮对话 + 文档管理
- [ ] W5：评估脚本 + 检索效果对比实验
- [ ] W6：Docker 化 + 文档收尾

## 检索效果对比（W5 补充实验数据）

| 方案            | 命中率 |
| ------------- | --- |
| 纯向量召回         | -   |
| 混合召回（向量+BM25） | -   |
| 混合召回 + Rerank | -   |
