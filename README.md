# RAG Knowledge Base —— 企业知识库智能问答系统

基于 **RAG（检索增强生成）** 的知识库问答系统。上传 PDF / Markdown / TXT / DOCX 文档后自动切分与向量化，
通过 **BM25 + 向量混合召回 → RRF 融合 → BGE-Rerank 精排** 三段式检索定位相关内容，
调用 DeepSeek 生成**带引用来源**的回答，SSE 流式输出，支持多轮对话与会话历史回看。

> 本项目是完整的工程实践，检索链路每一段都有量化评估数据支撑（见[检索效果对比](#检索效果对比)），
> 不是"能跑起来就行"的玩具 Demo。

---

## 目录

- [核心特性](#核心特性)
- [检索链路设计](#检索链路设计)
- [Agent 模式（工具调用）](#agent-模式工具调用)
- [MCP Server（对外暴露工具）](#mcp-server对外暴露工具)
- [缓存设计](#缓存设计)
- [Prompt 版本管理](#prompt-版本管理)
- [检索效果对比](#检索效果对比)
- [技术栈与选型理由](#技术栈与选型理由)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [如何验证](#如何验证)
- [环境变量](#环境变量)
- [API 文档](#api-文档)
- [工程实践](#工程实践)
- [常见问题](#常见问题)
- [后续演进](#后续演进)

---

## 核心特性

| 特性 | 实现说明 |
| :--- | :--- |
| **多格式文档解析** | PDF（pypdf，自动识别扫描件并报错）/ Markdown / TXT / DOCX，单文件上限 50 MB |
| **混合检索** | 向量召回（语义）+ BM25 召回（关键词精确匹配），RRF 融合解决两套分数不可比的问题 |
| **Cross-Encoder 精排** | 召回 20 条 → BGE-Reranker-v2-m3 精排取 5 条，前端展示的分数就是它的权威分 |
| **流式输出** | SSE 逐 token 推送，配 `X-Accel-Buffering: no` 防止反向代理缓冲导致"假流式" |
| **多轮对话** | LLM 查询改写做指代消解（"它怎么样" → 完整问题）；改写只用于检索，回答仍用原话 |
| **Agent 模式** | 模型自主决定「直接回答 / 调检索工具 / 调统计工具」，支持多轮工具调用，前端实时展示工具调用过程 |
| **MCP Server** | 同一套工具以 MCP 协议对外暴露，Claude Desktop / Cursor 等任意 MCP 客户端可直接查询本知识库 |
| **分层缓存** | 检索结果缓存（默认开）+ 答案缓存（默认关），LRU + TTL 双淘汰，知识库一变自动失效 |
| **Prompt 版本管理** | 提示词带版本号存于 Git，`extends` 继承只写差异，版本随答案落库可回溯，配错自动回退 |
| **引用溯源** | 每条回答附带引用片段（文件名 + 页码 + 原文截取），点击可定位到原文位置 |
| **会话持久化** | MySQL 存储会话/消息/文档元数据，外键级联删除 |
| **评估体系** | LLM 自动合成评测集 + Recall@5 / MRR 双指标，可复跑验证任何检索改动 |
| **容器化** | 一条 `docker compose up` 起 MySQL + 后端，healthcheck 保证依赖顺序 |
| **可观测性** | 结构化日志 + 请求中间件 + 慢请求告警 + 全局异常兜底 |

---

## 检索链路设计

这是整个项目的技术核心，面试建议重点准备这一段。

```
用户提问
   │
   ├─[多轮场景]─ 查询改写：LLM 做指代消解 → standalone_question
   │                    （只用于检索，回答仍用原始问题）
   │
   ▼
┌──────────────── 双路召回（求广）────────────────┐
│  向量召回 top20                 BM25 召回 top20  │
│  BGE-M3 embedding               jieba 中文分词   │
│  余弦相似度，懂语义              score>0 才返回   │
│  缺点：可能漏精确术语            缺点：不懂语义   │
└────────────────────┬────────────────────────────┘
                     ▼
            RRF 融合：score = Σ 1/(60 + rank)
            （只看排名不看分数 —— 两套分数体系不可直接相加）
                     ▼
            质量过滤：向量路要求 similarity ≥ 0.5
                     BM25 路直通（score>0 本身就是强信号）
                     ▼
            BGE-Reranker-v2-m3 精排 → top5（求准）
                     ▼
            DeepSeek 流式生成 + 引用来源
```

### 三个关键设计决策

**1. 为什么要混合召回，只用向量不行吗？**

向量检索擅长语义匹配，但对**专有名词、型号、缩写**容易失手（"厦航奖学金"被拆成语义相近但不同的概念）。
BM25 正好相反：它只认关键词是否真实出现，精确但不懂语义。两者互补。

**2. 为什么用 RRF 而不是把两套分数加权求和？**

余弦相似度的值域是 `0~1`，BM25 的分数**没有上界**（取决于词频和文档长度），两者量纲完全不同，
直接加权求和等于把不可比的数字硬凑在一起。RRF（Reciprocal Rank Fusion）**只看文档在各路结果中的排名**：

```python
rrf_score = Σ  1 / (60 + rank)     # 60 是平滑常数，防止头部排名权重过于悬殊
```

排名是跨检索器可比的序数，这就绕开了量纲问题。两路都命中的条目自然获得"双份"分数，自动排到前面。

**3. 有了混合召回为什么还要 Rerank？**

RRF 融合是**无监督**的启发式规则，它不知道哪条真的更相关。BM25 路召回的内容常常只是关键词碰上了，
实际答非所问 —— 这些噪声会污染 RRF 的排名。**实测数据证明了这一点**（详见下一节）：
混合召回单独使用反而比纯向量**更差**，加上 Rerank 之后才反超。

---

## Agent 模式（工具调用）

普通模式是**固定链路**：每次提问必检索 → 拼上下文 → 生成。哪怕问"你好"，也要付一次检索成本。

Agent 模式把「要不要检索」这个决定权交给模型，它自己判断该直接回答、还是调用工具。
通过接口参数 `agent=true` 切换，**默认行为不变**，老调用方完全不受影响。

```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"知识库里一共有几份文档？","agent":true}'
```

### 两个内置工具

| 工具 | 参数 | 返回 | 典型触发场景 |
| :--- | :--- | :--- | :--- |
| `search_knowledge_base` | `query`（必填） | 命中的文档片段 + 带编号的参考资料文本 | "BGE-M3 论文讲了什么"、"MySQL.md 里怎么建索引" |
| `get_knowledge_base_stats` | 无 | 文档总数、切片总数、文件清单 | "有几份文档"、"装了哪些文件" |

**检索工具内部完整复用现有 RAG 链路**（混合召回 → 相似度阈值过滤 → Rerank 精排），
不另写一套检索逻辑，因此 Agent 模式的答案质量与普通模式一致。

### 工作流程

```
用户提问
  │
  ├─ 载入最近 N 轮历史 + AGENT_SYSTEM_PROMPT
  │
  ▼
┌─────────────────────────────────┐
│ 调 LLM（带 tools 参数）          │
│ tool_choice="auto" → 模型自决    │
└─────────────────────────────────┘
  │
  ├─ 模型输出 tool_calls？
  │    ├─ 否 → 直接输出答案 → 流式推送 → 结束
  │    └─ 是 ↓
  │
  ├─ 执行工具 → 回填 role=tool 消息 → 再次调 LLM（最多 3 轮）
  │
  ▼
生成答案 + 引用来源 → 落库 → 结束
```

### 三个关键设计

**① 防死循环硬闸门**
`AGENT_MAX_TOOL_ROUNDS`（默认 3）控制最大工具调用轮数。达到上限后**不再把 `tools` 传给模型**——
没有工具可调，模型只能基于已有信息给出最终答案。这比"提示模型别乱调"可靠得多：
提示词约束是软约束，参数控制是硬约束。

**② OpenAI 协议的强制回填顺序**
assistant 的 `tool_calls` 消息必须先回填，再逐条回填 `role=tool` 的工具结果，
且每条工具结果的 `tool_call_id` 必须与对应的 `tool_calls[].id` 匹配，否则下一轮请求会被 API 拒绝。

**③ 流式与工具调用共用一次请求**
`delta.content` 直接转发前端（保住打字机体验），`delta.tool_calls` 按 `index` 分片累积拼接
（OpenAI 会把工具名和参数拆成多片推送）。因此"模型直接回答"时**不会多发一次请求**。

### 工具调用的实际表现

| 问题 | 模型选择的工具 | 结果 |
| :--- | :--- | :--- |
| 你好，今天天气怎么样？ | 无（直接回答） | 闲聊不触发检索，不浪费成本 |
| 知识库里一共有几份文档？ | `get_knowledge_base_stats` | 正确返回 7 份文档 / 462 个切片及完整清单 |
| BGE-M3 论文讲了什么？ | `search_knowledge_base` | 召回相关片段并标注引用编号 |
| MySQL.md 里怎么建索引？ | `search_knowledge_base` | 召回索引相关内容 |

工具执行失败时（如 embedding 服务不可用），错误会被转成**工具结果文本回填给模型**，
由模型决定换 query 重试还是如实说明——不会让整个问答请求 500。

### 新增的 SSE 帧

Agent 模式比普通模式多两种帧，前端据此展示"调用了什么工具"的过程：

```
data: {"type": "tool",        "name": "search_knowledge_base", "arguments": {"query":"RAG"}, "round": 1}
data: {"type": "tool_result", "name": "search_knowledge_base", "summary": "检索「RAG」：命中 5 条片段", "round": 1}
```

`tool` 帧在工具执行前推送（展示调用了什么、传了什么参数），
`tool_result` 帧在执行后推送（展示结果摘要）。普通模式的帧序列完全不变。

### 代码位置

```
app/agent/
├── __init__.py           # 包说明
├── tools.py              # 工具层：JSON Schema 定义 + 两个工具实现 + 统一分发入口
└── agent_service.py      # 决策层：工具调用循环、流式转发、防死循环闸门
```

---

## MCP Server（对外暴露工具）

Agent 模式解决的是"**我自己**的问答服务能调工具"。
如果想让 **Claude Desktop / Cursor / Cline** 这些外部 AI 应用也来查我们的知识库呢？
按老办法，每个应用的工具格式都不一样，得各写一套适配代码。

MCP（Model Context Protocol）就是为了消灭这件事：我只需要写**一个** server，
所有支持 MCP 的客户端都能直接连。它常被称为"AI 应用的 USB-C 接口"，
本质是把 **N×M 的适配问题**（N 个应用 × M 个数据源）降成 **N+M**。

```
                        ┌──────────────────────────┐
  自家前端  ──HTTP/SSE──▶│   app/agent  (任务1)      │
                        │   OpenAI tool_calling    │
                        └────────────┬─────────────┘
                                     │ 同一个 run_tool
外部客户端 ──stdio/JSON-RPC──▶┌──────┴──────────────┐
  Claude Desktop / Cursor    │  mcp_server (任务2) │
                             └─────────────────────┘
```

**关键：两条路调用的是同一个 `app/agent/tools.py` 里的 `run_tool`。**
业务逻辑只有一份，MCP server 只是一层协议外壳，没有复制任何检索代码。

### 启动

```bash
# 独立 stdio 进程，由 MCP 客户端拉起，也可以手动跑起来看日志
python -m mcp_server.server

# 冒烟测试：用真实 MCP 客户端连真实 server，验证握手 / 工具发现 / 工具调用
python -m tests.test_mcp_server
```

### 客户端配置

**Claude Desktop**（`claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "D:/rag-knowledge-base/.venv/Scripts/python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "D:/rag-knowledge-base"
    }
  }
}
```

**Cursor**（`.cursor/mcp.json`，项目级）：

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "D:/rag-knowledge-base/.venv/Scripts/python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "D:/rag-knowledge-base"
    }
  }
}
```

> 必须把 `command` 指向装了依赖的那个 `python.exe`（虚拟环境里的），
> 用系统 Python 会报 `ModuleNotFoundError: No module named 'mcp'`。

### 暴露的工具

| 工具 | 参数 | 说明 |
| :--- | :--- | :--- |
| `search_knowledge_base` | `query` | 检索知识库正文，返回带编号的参考资料文本 |
| `get_knowledge_base_stats` | 无 | 文档总数、切片总数、文件清单 |

### 两个 stdio 特有的坑

**① stdout 只能给协议用**
stdio 传输下客户端通过进程的**标准输出**读 JSON-RPC 报文，
业务代码里任何一句 `print` 都会插进这条流。项目里 `embedder.py` 就有
`print("embedding 第 x/y 批...")`，实测不加处理时客户端直接报：

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb5 in position 12
```

（中文 print 按 GBK 写进 stdout，客户端按 UTF-8 解码失败。）
解法：工具执行期间用 `contextlib.redirect_stdout(sys.stderr)` 把 print 引到 stderr，
**执行完立即恢复**，这样不会影响 MCP SDK 自己往 stdout 写报文。

**② `.env` 加载依赖工作目录**
客户端拉起进程时 cwd 是不确定的，`app/config.py` 里无参数的 `load_dotenv()` 会找不到 `.env`，
导致所有 API key 为空。解法：在 `import app.*` **之前**，
用 `Path(__file__).resolve()` 算出项目根并显式 `load_dotenv(PROJECT_ROOT / ".env")`。

### 代码位置

```
mcp_server/
├── __init__.py           # 包说明：与 app/agent 的分工
└── server.py             # MCPServer 实例 + 两个工具 + stdio 入口
tests/test_mcp_server.py  # 冒烟测试：真实客户端连真实 server
```

---

## 缓存设计

一次检索的耗时几乎全在两次网络 API 上（预热后实测）：

```
embedding API : 约 3.9 秒
rerank API    : 约 3.9 秒
本地计算       : 可忽略
```

所以缓存省的不是 CPU，是**两次跨网络往返**。命中后延迟降到 0.1 毫秒量级。

### 缓存什么

| 层级 | 缓存内容 | 默认 | 省掉什么 |
| :--- | :--- | :--- | :--- |
| **检索结果** | 「召回 → 过滤 → 精排」的最终结果 | 开 | embedding + rerank 两次 API |
| **答案** | LLM 生成的完整答案 + 引用来源 | 关 | 上面两次 + 整次生成 |

缓存挂在**精排之后**而不是召回之后：只缓存召回结果的话，rerank 那次 API 还是省不掉，只能省一半。

答案缓存默认关闭的原因：生成有随机性（`temperature=0.3`）、回放不是真流式、
且只在「无历史」时启用（有历史时同一问题的答案依赖上下文，缓存会串台）。

### 三个关键设计

**① LRU 和 TTL 两个都要**
TTL（默认 300 秒）管"新鲜度"，LRU（默认 500 条）管"容量"。
只靠 TTL 的话，5 分钟内问一万个不同问题照样撑爆内存——
TTL 防的是时间上的无限增长，LRU 防的是数量上的无限增长，两者互补。

**② 知识库指纹：缓存最怕的不是慢，是脏**
上传新文档后同一个问题的正确答案变了，如果 key 只有问题本身，用户会一直拿到旧答案。
所以 key 里带上知识库指纹：

```python
key    = md5(f"{scope}|{归一化问题}|kb={指纹}")
指纹   = f"{文档数}:{切片总数}:{最大文档id}"
```

指纹一变 → key 一变 → 旧条目自然读不到，**等价于自动失效，不用人工清理**。

**③ 指纹查不到时禁用缓存（降级：宁可慢，不能错）**
指纹要查 MySQL，MySQL 挂了就不知道知识库有没有变。
此时**既不读也不写缓存**，直接走真实检索——返回过期答案比慢几秒严重得多。
统计里单独记了 `disabled` 次数方便监控。

### 实测数据

```bash
python -m scripts.bench_cache           # 默认取评测集 30 条
python -m scripts.bench_cache --n 10    # 小样本，跑得快
python -m scripts.bench_cache --no-cache # 对照组：全程关缓存
```

压测方法：同一个问题问两遍——第一遍缓存里没有（真实走 API），第二遍命中。
**先跑一条 warmup**：jieba 加载 + Chroma 初始化的冷启动开销有 8 秒左右，
不预热的话第一轮数字全被污染，收益会虚高到十几万倍，那是自己骗自己。

`--n 10` 实测（硅基流动 API 当时约 3.9 秒/次，数字会随网络与额度波动）：

| 轮次 | 条数 | 总耗时 | 平均延迟 | P95 | 命中 | 命中率 |
| :--- | ---: | -----: | -------: | ---: | ---: | -----: |
| 冷缓存（第 1 遍） | 10 | 77.9s | 7786.4ms | 7927.0ms | 0 | 0% |
| 热缓存（第 2 遍） | 10 | 0.0s | 0.3ms | 0.0ms | 10 | 100% |

两轮合计：开缓存 77.9s，不开缓存 155.7s，**API 调用从 20 次降到 10 次**。

> ⚠️ 踩过的坑：第一次跑 30 条时命中率是 **0%**——30 条真实检索要 342 秒，
> 而 TTL 是 300 秒，第一轮还没跑完最早写入的条目就过期了。
> 这不是缓存没用，是**压测耗时超过了缓存有效期**。
> 所以压测脚本默认把 TTL 放大到 3600 秒（可用 `--ttl` 覆盖）。

开答案缓存后整轮问答的对比（同一个问题第二次提问）：

```
第 1 次：session → 53 个 token 帧 → sources → done    11932 ms
第 2 次：session → cache(检索) → cache(答案) → token → sources → done   24 ms
```

### 怎么关（一键回滚）

```bash
CACHE_ENABLED=false   # 所有缓存逻辑直接短路，行为与加缓存前完全一致
```

因为这个开关的存在，加缓存后**不需要重跑评测集**：
`app/rag/` 一行没动，评测脚本走的还是原路径，指标不受影响。

### 代码位置

```
app/cache/
├── __init__.py          # 包说明：三条设计原则
└── kb_cache.py          # TTL+LRU 缓存、命中率统计、知识库指纹
app/services/retrieval_service.py  # 检索编排（召回+过滤+精排），缓存挂在这里
scripts/bench_cache.py   # 压测脚本
```

命中率可以在健康检查里直接看：

```bash
curl localhost:8000/api/health | jq .cache
# {"enabled": true, "retrieval": {"hits": 30, "misses": 30, "hit_rate": 0.5, ...}}
```

### 为什么不用 Redis

单机部署（Docker 单容器 + 本地 Chroma），知识库规模十万级以内，
进程内缓存是纳秒级，Redis 还要多一次网络往返（0.3~1ms）和多运维一个服务。

**代价要说清楚**：多副本部署时各副本缓存不共享，命中率会被副本数稀释。
到时候换 Redis 即可——接口就是 `get/set/clear`，替换成本很低。

---

## Prompt 版本管理

### 为什么要版本化

三个改 prompt 时必然遇到的痛点：

1. **改完就覆盖，无从回溯**——线上答案变差了，不知道是哪次改动引起的，只能翻 Git log 猜
2. **没法灰度、没法快速回滚**——想试新 prompt 只能改代码重启，试错了再改回去
3. **改好改坏只能凭感觉**——检索指标（Recall/MRR）对 prompt 完全不敏感，帮不上忙

对应做法：模板抽成带版本的配置文件 → 版本号随答案落库 → 提供对比脚本。

### 目录结构

```
app/core/prompt_versions/
├── v1.json    # 基线版本：从改造前的 prompts.py 原样迁出，逐字未改
└── v2.json    # extends v1，只写「跟 v1 不一样的那一个模板」
```

版本之间用 `extends` 继承：v2 只覆盖主问答 prompt，其余四个模板自动沿用 v1，
不用复制粘贴整份模板。改 prompt 从此只改一处差异。

### 怎么加一个新版本

新建 `app/core/prompt_versions/v3.json`：

```json
{
  "version": "v3",
  "name": "这次改了什么",
  "extends": "v2",
  "changes": ["逐条列出变更，方便日后回溯"],
  "templates": { "rag_system": "…只写需要改的那个模板…" }
}
```

模板里允许的变量只有 `context / history / question / index / filename / page_info / content`，
**加载时就会校验**，写错占位符立刻报错，不会等到线上某次问答才 KeyError。

### 怎么切换和回滚

```bash
PROMPT_VERSION=v2 uvicorn app.main:app     # 切到 v2
PROMPT_VERSION=v1 uvicorn app.main:app     # 回滚，改一个环境变量即可
```

配错版本号（或 JSON 写坏）时自动回退 v1 并告警，不让 prompt 问题拖垮整个服务。

### 每次回答都记了版本

`chat_messages` 表有 `prompt_version` 列，每条 assistant 消息都记下生成时用的版本：

```sql
SELECT role, prompt_version, created_at FROM chat_messages WHERE role='assistant';
```

前端也会在回答上方显示 `prompt v1` 徽标。于是"这条答案答得不好"可以精确归因到某个版本。

> 老库升级：`init_db()` 里带幂等的 `ALTER TABLE`，服务重启自动补列，不需要手工迁移。
> 也可以用 `python -c "from app.core.database import ensure_prompt_version_column; ensure_prompt_version_column()"` 单独执行。

### 改好还是改差，怎么判断

检索指标衡量不了生成质量，所以对比脚本做的是**控制变量 + 人工判断**：
同一批检索结果喂给两个版本，唯一变量是 prompt。

```bash
python -m scripts.compare_prompts --n 3        # 对比 v1 vs v2
```

实测（2 条问题，样本小，仅作示例）：

| 版本 | 平均长度 | 平均引用标注 |
| :--- | ---: | ---: |
| v1 | 108 字 | 1.5 处 |
| v2 | 147 字 | **4.5 处** |

v2 要求"每个关键结论后都标注来源"，标注密度从 1.5 涨到 4.5，说明改动确实生效了。
但标注变多**不等于质量更好**——可能是溯源更细，也可能是啰嗦，最终要人来看答案。

### 三个设计取舍

| 取舍 | 选择 | 理由 |
| :--- | :--- | :--- |
| 存哪里 | 文件（Git）而非数据库 | prompt 要跟代码一起评审、一起 diff；放数据库等于放进了最难审计的地方 |
| 何时加载 | 启动时加载，改版本要重启 | prompt 属于发布配置。允许运行时热改，会出现"同一问题前后风格不一致" |
| 默认行为 | v1 与改造前逐字一致 | `app/rag/` 下的生成代码一行没动，只是模板搬了家 |

### 代码位置

```
app/core/prompt_registry.py        # 版本加载 / extends 继承 / 占位符校验 / 安全渲染
app/core/prompt_versions/*.json    # 版本文件
app/core/prompts.py                # 兼容层：把默认版本暴露成同名常量
```

---

## 检索效果对比

评测集：由 LLM 从真实文档切片自动合成的 **92 条**问答对（`data/eval_set.json`），
每条包含问题、期望命中的关键词、来源文档 id。

| 方案 | Recall@5 | MRR | 说明 |
| :--- | :---: | :---: | :--- |
| A 纯向量召回 | 88.0% | 0.766 | 基线方案 |
| B 混合召回（仅 RRF 融合） | 82.6% | 0.702 | **比基线还差** |
| C 混合召回 + Rerank | **90.2%** | **0.815** | 最终采用 |

> **Recall@5**：正确答案是否出现在前 5 条里（有=1，无=0），取平均。
> **MRR**（Mean Reciprocal Rank）：正确答案排第几位取倒数（第 1 名=1 分，第 2 名=0.5，第 5 名=0.2），衡量"好结果排得有多靠前"。

### 这个实验结果回答了一个面试高频问题

**"加了 BM25 反而变差了，说明 BM25 没用吗？"** —— 恰恰相反，说明**混合召回必须搭配精排**才有价值。

原因拆解：BM25 路召回了一批"关键词碰巧重叠"的噪声片段。在纯 RRF 排序下，这些噪声凭借自身排名往前提，
挤占了真正相关的向量结果的名额，导致 Recall@5 从 88.0% 掉到 82.6%。
而 Rerank（cross-encoder）能把 query 和候选片段**拼在一起做深度交互打分**，识别出这些噪声并压到后面 ——
加了它之后 Recall@5 提升到 90.2%，MRR 也从 0.766 提升到 0.815。

**结论**：召回阶段要"广"（多路并行，宁可多召回），排序阶段要"准"（用更强的模型精排）。两者职责分离，缺一不可。

复跑命令：

```bash
python -m tests.evaluate           # 只看汇总
python -m tests.evaluate --detail  # 打印每题命中情况
```

重新生成评测集：

```bash
python -m scripts.build_eval_set   # 每篇文档追加合成到 10 条，增量合并已有结果
```

---

## 技术栈与选型理由

| 组件 | 选型 | 为什么选它 |
| :--- | :--- | :--- |
| Web 框架 | FastAPI | 原生异步 + 自动 OpenAPI 文档 + Pydantic 校验，`StreamingResponse` 天然支持 SSE |
| 文本切分 | LangChain `RecursiveCharacterTextSplitter` | 按字符优先级递归切分（段落→句子→词），比定长切分更能保住语义完整性 |
| 向量库 | Chroma | 本地嵌入式、零运维、持久化到磁盘，适合中小规模（十万级以内）场景 |
| 关键词检索 | rank-bm25 + jieba | 纯 Python 实现无外部服务依赖；BM25 需要分词，中文用 jieba |
| Embedding | BGE-M3（硅基流动 API） | 中文语义表征强，支持稠密+稀疏+多向量三种检索模式，有免费额度 |
| Rerank | BGE-Reranker-v2-m3（硅基流动 API） | 中文 cross-encoder 里性价比高，不需要本地 GPU |
| 生成模型 | DeepSeek | 兼容 OpenAI SDK（`openai` 包换 `base_url` 即可），成本低、中文能力强 |
| 元数据库 | MySQL 8.x + SQLAlchemy 2.0 | 文档/会话/消息是强结构化关系数据；用 ORM 而非裸 SQL，避免拼接注入 |
| 工具协议 | MCP（官方 Python SDK `mcp`） | 一份 server 对接所有 MCP 客户端，把 N×M 适配降成 N+M；stdio 传输，零端口占用 |
| 容器编排 | Docker Compose | 一条命令拉起全部依赖，healthcheck 解决服务启动顺序问题 |

**为什么不用 PostgreSQL + pgvector？**
向量数据和业务元数据规模都不大，引入 pgvector 会增加运维复杂度和学习成本。这里用 MySQL 管关系数据、
Chroma 管向量，**各司其职、职责清晰**。真到千万级规模再考虑迁移，届时只需要替换 `app/rag/vector_store.py` 的实现。

---

## 项目结构

```
rag-knowledge-base/
├── app/
│   ├── main.py                    # 应用入口：路由注册、静态文件托管、日志中间件、全局异常处理
│   ├── config.py                  # 集中读取环境变量（禁止代码里散落 os.getenv）
│   ├── api/
│   │   ├── documents.py           # 文档上传 / 列表 / 删除
│   │   └── chat.py                # SSE 问答 + 会话增删改查
│   ├── core/
│   │   ├── database.py            # 引擎、SessionLocal、init_db
│   │   ├── logging.py             # RotatingFileHandler + 第三方库降噪
│   │   ├── time_utils.py          # utc_now() / ensure_utc()，统一时间基准
│   │   ├── prompts.py             # 提示词兼容层：把默认版本暴露成同名常量
│   │   ├── prompt_registry.py     # 版本加载 / extends 继承 / 占位符校验
│   │   └── prompt_versions/       # v1.json（基线）、v2.json（extends v1）
│   ├── models/
│   │   ├── db_models.py           # ORM 模型：documents / chat_sessions / chat_messages
│   │   └── schemas.py             # Pydantic 请求响应模型
│   ├── rag/
│   │   ├── loader.py              # 多格式解析（含扫描件 PDF 检测）
│   │   ├── splitter.py            # 递归字符切分
│   │   ├── embedder.py            # BGE-M3 embedding（带失败重试）
│   │   ├── vector_store.py        # Chroma 封装
│   │   ├── bm25_store.py          # BM25 索引（增删后自动重建）
│   │   ├── retriever.py           # 双路召回 + RRF 融合
│   │   ├── reranker.py            # cross-encoder 精排
│   │   └── generator.py           # 流式生成 + 查询改写
│   ├── agent/
│   │   ├── tools.py               # 工具层：JSON Schema + 检索/统计工具 + 统一分发
│   │   └── agent_service.py       # 决策层：工具调用循环 + 防死循环闸门 + 流式转发
│   └── services/
│       ├── chat_service.py        # 问答主链路编排（0→8 步）
│       ├── session_service.py     # 会话 DAO（含事务边界）
│       ├── retrieval_service.py   # 检索编排（召回+过滤+精排），缓存挂在这里
│       ├── document_service.py    # 入库业务编排
│       └── document_repository.py # 文档元数据 DAO
├── frontend/index.html            # 单文件前端（会话侧边栏 + 流式渲染 + 引用面板）
├── mcp_server/
│   ├── __init__.py                # 包说明：与 app/agent 的分工
│   └── server.py                  # MCPServer 实例 + 工具注册 + stdio 入口
├── app/cache/
│   ├── __init__.py                # 包说明：缓存层三条设计原则
│   └── kb_cache.py                # TTL+LRU 缓存、命中率统计、知识库指纹
├── tests/
│   ├── evaluate.py                # Recall@5 / MRR 评估脚本
│   └── test_mcp_server.py         # MCP 冒烟测试：真实客户端连真实 server
├── scripts/                       # 开发运维脚本（不入生产镜像）
│   ├── start_mysql.bat            # 一键启动本机 MySQL（自动申请管理员权限）
│   ├── build_eval_set.py          # LLM 合成评测集
│   ├── check_time.py              # 时间链路自检（验证存储确为 UTC）
│   ├── check_vector.py            # 向量库 / MySQL / 磁盘 三处数据一致性对账
│   ├── bench_cache.py             # 缓存压测：冷/热两轮对比，出命中率与加速比
│   ├── verify_agent.py            # Agent 验证：工具选择 / 防死循环闸门 / 检索不降质
│   ├── verify_prompts.py          # Prompt 版本验证：含「默认版本逐字一致」回归锚点
│   └── compare_prompts.py         # 两个 prompt 版本的答案对比（控制变量，只差 prompt）
│   ├── check_db.py                # 数据库表内容速览
│   ├── fix_legacy_time.py         # 历史时间偏移订正（带 dry-run）
│   ├── init_mysql.py              # 建库
│   ├── migrate_documents.py       # JSON 元数据迁移到 MySQL
│   ├── test_multi_turn.py         # 多轮对话测试
│   └── test_rewrite.py            # 查询改写测试
├── data/                          # 上传文档 + Chroma 持久化 + 评测集
├── logs/                          # 运行日志（按 5MB×5 滚动）
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 快速开始

### 方式一：Docker 一键启动（推荐）

```bash
# 1. 配置密钥
cp .env.example .env      # Windows 用 copy 或手动复制
#    编辑 .env 填入 DeepSeek / 硅基流动的 API Key

# 2. 起服务
docker compose up -d --build

# 3. 验证
curl http://127.0.0.1:8000/api/health
# {"status":"ok"}

# 浏览器打开 http://localhost:8000 即可使用
# 接口文档：http://localhost:8000/docs
```

> 国内拉取镜像慢？给 Docker Desktop 配 registry mirror（`daemon.json`）：
> ```json
> { "registry-mirrors": ["https://docker.m.daocloud.io", "https://docker.1ms.run"] }
> ```
> Dockerfile 内 pip 也已走清华源，如需切换：`docker build --build-arg PIP_INDEX=https://pypi.org/simple .`

停止：

```bash
docker compose down        # 停止容器（mysql_data 数据卷保留）
docker compose down -v     # 连数据卷一起删除
```

### 方式二：本地开发

```bash
# 1. 虚拟环境
python -m venv .venv
.venv\Scripts\activate     # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. 准备 MySQL
#    建库：CREATE DATABASE rag_kb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
#    然后把连接串填进 .env 的 DATABASE_URL

# 3. 启动
cp .env.example .env       # 编辑填入密钥和数据库密码
uvicorn app.main:app --reload --port 8000
```

应用启动时会自动建表（`init_db()` 幂等，已有表不会被覆盖）。

> ⚠️ 本机已装 MySQL 的话注意端口：compose 把容器 MySQL 映射到宿主机 **3307**，避免和本机 3306 冲突。

---

## 如何验证

每个能力都配了可复现的验证入口，不用靠肉眼判断「好像变快了」。

| 想验证什么 | 命令 | 通过标准 |
| :--- | :--- | :--- |
| Agent 会不会选工具 | `python -m scripts.verify_agent` | 工具选择 4/4；闸门轮数卡在 3；两条检索路径结果一致 |
| 同上（省额度版） | `python -m scripts.verify_agent --quick` | 只跑"检索质量不降"，不调 LLM |
| MCP Server 能不能被客户端用 | `python -m tests.test_mcp_server` | 握手成功 + 发现 2 个工具 + 两次调用都返回数据 |
| 缓存到底省了多少 | `python -m scripts.bench_cache --n 10` | 第 2 轮命中率 100%、平均延迟降到 1ms 以内 |
| Prompt 版本机制正常 | `python -m scripts.verify_prompts` | 6 组全 PASS，其中「默认版本与改造前逐字一致」是回归锚点 |
| 两个版本的答案差异 | `python -m scripts.compare_prompts --n 3` | 同一批检索结果下并排输出，看引用密度等指标变化 |
| 缓存（对照组） | `python -m scripts.bench_cache --n 10 --no-cache` | 两轮耗时接近，用来反证上面的差距确实来自缓存 |
| 检索质量有没有退步 | `python -m tests.evaluate` | Recall@5 / MRR 与历史基线一致（92 条，约 13 分钟） |

浏览器里也能直接看到：勾上「Agent 模式」后，回答上方会出现工具调用过程面板；
命中缓存时回答下方会显示「缓存命中」徽标。

```bash
curl http://127.0.0.1:8000/api/health   # 顺便返回缓存实时命中率
```

---

## 环境变量

复制 `.env.example` 为 `.env` 后按需修改。所有变量在 `app/config.py` 中集中读取并有默认值。

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `DEEPSEEK_API_KEY` | — | **必填**，DeepSeek 密钥 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 兼容 OpenAI 协议，换其他厂商改这里 |
| `LLM_MODEL` | `deepseek-chat` | 生成模型名 |
| `SILICONFLOW_API_KEY` | — | **必填**，硅基流动密钥（Embedding/Rerank） |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` | 硅基流动端点 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding 模型 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Rerank 模型 |
| `CHUNK_SIZE` | `500` | 切片字符数 |
| `CHUNK_OVERLAP` | `50` | 相邻切片重叠字符数（防止切断句子丢失上下文） |
| `VECTOR_TOP_K` | `20` | 向量路召回条数 |
| `BM25_TOP_K` | `20` | BM25 路召回条数 |
| `RERANK_TOP_K` | `5` | 精排后最终保留条数（也是喂给 LLM 的上下文条数） |
| `SIMILARITY_THRESHOLD` | `0.5` | 向量相似度阈值，低于视为无关召回 |
| `DATABASE_URL` | `mysql+pymysql://root:password@localhost:3306/rag_kb?charset=utf8mb4` | 数据库连接串 |
| `DATA_DIR` | `data` | 上传文件存放目录 |
| `CHROMA_DIR` | `data/chroma` | Chroma 持久化目录 |
| `HISTORY_TURNS` | `5` | 多轮对话携带的历史轮数 |
| `AGENT_MAX_TOOL_ROUNDS` | `3` | Agent 模式最大工具调用轮数（防死循环硬闸门） |
| `MCP_TRANSPORT` | `stdio` | MCP Server 传输方式，客户端直连时用 stdio |
| `PROMPT_VERSION` | `v1` | 默认使用的 prompt 版本，对应 `app/core/prompt_versions/<version>.json` |
| `CACHE_ENABLED` | `true` | 缓存总开关，关掉后行为与加缓存前完全一致（可一键回滚） |
| `CACHE_TTL` | `300` | 单条缓存存活秒数 |
| `CACHE_MAX_ENTRIES` | `500` | 缓存最多存多少条，超出按 LRU 淘汰 |
| `KB_FINGERPRINT_TTL` | `10` | 知识库指纹的本地缓存秒数（避免每次检索都查库） |
| `CACHE_ANSWER_ENABLED` | `false` | 答案缓存开关，默认关（生成有随机性，且回放非真流式） |
| `MYSQL_ROOT_PASSWORD` | `password` | 仅 Docker 部署用，compose 里的默认值 |

> ⚠️ `.env` 已在 `.gitignore` 和 `.dockerignore` 中排除，**切勿提交密钥到仓库**。

---

## API 文档

启动后访问 `http://localhost:8000/docs` 查看交互式 Swagger 文档。

### 文档管理

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| `POST` | `/api/documents/upload` | 上传文档，同步完成解析→切分→向量化→入库 |
| `GET` | `/api/documents` | 文档列表 |
| `DELETE` | `/api/documents/{id}` | 删除文档（向量库 + MySQL + 磁盘三处同步清理） |

### 问答与会话

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| `POST` | `/api/chat/stream` | **SSE 流式问答**，见下方帧格式 |
| `POST` | `/api/chat/sessions` | 新建会话 |
| `GET` | `/api/chat/sessions` | 会话列表（按创建时间倒序） |
| `GET` | `/api/chat/sessions/{id}/messages` | 历史消息（回看旧会话，含引用来源） |
| `DELETE` | `/api/chat/sessions/{id}` | 删除会话（消息级联删除） |

`POST /api/chat/stream` 请求体：

| 字段 | 类型 | 默认 | 说明 |
| :--- | :--- | :--- | :--- |
| `question` | string | 必填 | 用户问题 |
| `session_id` | int \| null | null | 为空表示新建会话，首帧返回新 id |
| `agent` | bool | **false** | 是否启用 Agent 模式（模型自主决定是否调用工具） |

### SSE 帧格式约定

普通模式（`agent` 不传或 false）：

```
data: {"type": "session", "id": 1}

data: {"type": "token",   "content": "回答的一个片段"}

data: {"type": "meta",     "prompt_version": "v1"}

data: {"type": "sources", "content": [{"index":1,"filename":"xx.pdf","page":3,"score":0.87,...}]}

data: {"type": "done"}
```

缓存命中时还会在最前面多一帧，前端据此显示徽标：

```
data: {"type": "cache", "layer": "retrieval", "hit": true}
```

Agent 模式（`agent=true`）在上述基础上**多两种帧**，用于展示工具调用过程：

```
data: {"type": "tool",        "name": "search_knowledge_base", "arguments": {"query":"RAG"}, "round": 1}

data: {"type": "tool_result", "name": "search_knowledge_base", "summary": "检索「RAG」：命中 5 条片段", "round": 1}
```

前端依次处理：`session` 记录会话 id → `tool` / `tool_result` 渲染工具调用过程 →
`meta` 显示 prompt 版本徽标 →
`token` 追加渲染 → `sources` 渲染引用面板 → `done` 结束加载态。

命中缓存时还会**多一种帧**（在 token 之前到达，前端据此显示"⚡ 缓存命中"徽标）：

```
data: {"type": "cache", "layer": "retrieval", "hit": true}
data: {"type": "cache", "layer": "answer",    "hit": true}
```

`layer=retrieval` 表示省掉了 embedding + rerank，`layer=answer` 表示连生成都省了。

### 健康检查

`GET /api/health`，被 docker-compose healthcheck 和部署脚本使用。顺带返回缓存命中率：

```json
{
  "status": "ok",
  "cache": {
    "enabled": true,
    "answer_enabled": false,
    "retrieval": {"hits": 30, "misses": 30, "total": 60, "hit_rate": 0.5, "evictions": 0, "expirations": 0, "disabled": 0},
    "answer": {"hits": 0, "misses": 0, "total": 0, "hit_rate": 0.0, "evictions": 0, "expirations": 0, "disabled": 0}
  }
}
```

---

## 工程实践

这一节记录的是"代码之外"的东西，面试时能把它们讲清楚是很加分的。

### 1. 时间处理：统一存 UTC

**约定**：数据库一律存 UTC，输出时带时区标记（`...Z`），由前端按本地时区渲染。

- 写入：`utc_now()` = `datetime.now(timezone.utc).replace(tzinfo=None)`
  —— 返回 **naive** UTC，因为 MySQL 的 `DATETIME` 列不存时区，带 `tzinfo` 的对象交给驱动时行为因驱动而异
- 读出：`ensure_utc()` 给库里的 naive 值补回 `tzinfo=utc`，Pydantic 才会序列化成带 `Z` 的 ISO8601
- 前端：`new Date(iso).toLocaleString("zh-CN", {hour12: false})`

**为什么不用 `TIMESTAMP` 列？** 它有 2038 年上限，且隐含的时区转换会让行为难以预测。
**为什么不用 `datetime.utcnow()`？** Python 3.12 起已废弃，且它返回 naive UTC，语义含糊。

验证工具（会自动用**磁盘文件 mtime** 校准，判断库里存的到底是不是 UTC）：

```bash
python -m scripts.check_time
```

### 2. 数据迁移的可验证性

`scripts/fix_legacy_time.py` 校准历史时间时，没有用数据自身去判断数据是否正确，而是引入**外部事实源**：
操作系统写入的文件 `mtime` 必然是本地墙钟，用它来对比数据库中的 `created_at`：

```
diff = 数据库 created_at − 磁盘 mtime
diff ≈  0h  →  当时写的是本地墙钟
diff ≈ −8h  →  当时写的已是 UTC
```

> 教训：初版曾用"最大值是否为未来时间"判断，结果对几天前的数据**误判**。
> 任何"在某个条件下碰巧成立"的启发式判据，换个数据集就会失效 —— 迁移脚本必须带 dry-run 和订正后自动复查。

### 3. 可观测性

- 日志滚动切割（5MB × 5 个备份），避免磁盘被撑爆
- 请求中间件记录 `方法 路径 状态码 耗时`，>3 秒的请求标 WARNING
- 全局异常兜底：未捕获异常返回统一 JSON 并写入完整 traceback
- `mute_noisy()` 压制 jieba / httpx / chromadb 等第三方库的 DEBUG 噪声

### 4. 依赖健壮性

`depends_on` 默认**只保证容器进程启动**，不保证服务可用。这里用 healthcheck + `condition: service_healthy`，
确保 MySQL 真正能接受连接后才启动后端，避免"启动竞态"。

### 5. 数据一致性

增删文档后向量库、MySQL、磁盘三处可能出现不一致（测试时真实踩过）。
`scripts/check_vector.py` 提供三方自动对账：

```bash
python -m scripts.check_vector
```

---

## 常见问题

<details>
<summary><b>Docker 报 "Virtualization support not detected"</b></summary>

BIOS 里开了虚拟化 ≠ Windows 加载了 hypervisor。完整依赖链是 **CPU VT-x → Windows 功能 → WSL2 → Docker**，
需逐层排查：

```powershell
dism.exe /Online /Enable-Feature /FeatureName:Microsoft-Windows-Subsystem-Linux /All /NoRestart
dism.exe /Online /Enable-Feature /FeatureName:HypervisorPlatform /All /NoRestart
bcdedit /set hypervisorlaunchtype auto
# 重启后
wsl --update
wsl --set-default-version 2
```

若仍失败，检查「Windows 安全中心 → 设备安全性 → 内核隔离 → 存储器完整性」是否与虚拟化冲突。
</details>

<details>
<summary><b>MySQL 连不上，报 ERROR 2003 (HY000) / Can't connect</b></summary>

错误码 2003/10061 = **服务没启动**（传输层被拒绝），不是密码错误（那是 1045）。Windows 上：

```powershell
net start MySQL80        # 需管理员权限
```

本项目提供一键脚本：`scripts/start_mysql.bat`（自动申请管理员权限并等待端口就绪）。
</details>

<details>
<summary><b>脚本报 ModuleNotFoundError: No module named 'app'</b></summary>

必须用**模块方式**运行，不能用文件路径：

```bash
python -m scripts.check_time      # 正确
python scripts/check_time.py      # 错误
```

原因：Python 会把脚本所在目录塞进 `sys.path[0]`，直接跑脚本时 `scripts/` 成了根目录，找不到 `app` 包。
</details>

<details>
<summary><b>日志里中文乱码</b></summary>

PowerShell 默认 GBK 码页。先切换：

```powershell
chcp 65001
```

Git Bash / VSCode 终端不受影响。
</details>

<details>
<summary><b>PowerShell 里 grep 报错</b></summary>

PowerShell 没有 `grep`，用 `findstr`：

```powershell
netstat -ano | findstr ":3306"
```
</details>

<details>
<summary><b>上传 PDF 提示解析失败</b></summary>

扫描件（图片型 PDF）没有文本层，pypdf 提取不到内容会主动报错并返回 400。这是**预期行为**，需要先做 OCR。
</details>

---

## 后续演进

- [ ] **异步入库**：文档量大时上传会阻塞，改用 Celery / ARQ 任务队列 + 轮询入库状态
- [ ] **表格与图片理解**：PDF 中的表格目前会被拆散，考虑接入版面分析（如 PaddleOCR + 多模态模型）
- [ ] **RAG-Fusion**：多路查询改写并行召回，进一步提升召回率
- [ ] **上下文压缩**：长对话时对历史做摘要压缩，突破上下文窗口限制
- [ ] **用户体系**：多租户隔离 + 文档权限控制
- [ ] **向量库迁移**：数据量到百万级后从 Chroma 迁到 Milvus / Qdrant
- [ ] **答案质量评估**：接入 RAGAS 做忠实度（Faithfulness）与答案相关性评估，不只评估检索环节
- [x] **MCP Server 封装**：`mcp_server/` 已把 Agent 工具封装成 MCP Server，Claude Desktop / Cursor 可直接调用本知识库
- [x] **缓存层**：检索结果缓存（默认开）+ 答案缓存（默认关），LRU+TTL 双淘汰，知识库指纹自动失效，含压测脚本
- [ ] **相似 query 缓存匹配**：当前是精确匹配，语义相似匹配需要向量近邻检索，性价比待评估
- [x] **Prompt 版本管理**：提示词抽成带版本号的 JSON（`extends` 继承，只写差异），版本号随答案落库可回溯，配错自动回退 v1，含对比脚本
- [ ] **结构化输出**：用 JSON Schema 约束模型输出（答案正文 + 引用列表），前端按结构化结果渲染

---

## License

MIT
