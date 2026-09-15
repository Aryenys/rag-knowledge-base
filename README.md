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
- [检索效果对比](#检索效果对比)
- [技术栈与选型理由](#技术栈与选型理由)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
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
│   │   └── prompts.py             # 提示词集中管理
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
│   └── services/
│       ├── chat_service.py        # 问答主链路编排（0→8 步）
│       ├── session_service.py     # 会话 DAO（含事务边界）
│       ├── document_service.py    # 入库业务编排
│       └── document_repository.py # 文档元数据 DAO
├── frontend/index.html            # 单文件前端（会话侧边栏 + 流式渲染 + 引用面板）
├── tests/evaluate.py              # Recall@5 / MRR 评估脚本
├── scripts/                       # 开发运维脚本（不入生产镜像）
│   ├── start_mysql.bat            # 一键启动本机 MySQL（自动申请管理员权限）
│   ├── build_eval_set.py          # LLM 合成评测集
│   ├── check_time.py              # 时间链路自检（验证存储确为 UTC）
│   ├── check_vector.py            # 向量库 / MySQL / 磁盘 三处数据一致性对账
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

### SSE 帧格式约定

```
data: {"type": "token",   "content": "回答的一个片段"}

data: {"type": "sources", "content": [{"index":1,"filename":"xx.pdf","page":3,"score":0.87,...}]}

data: {"type": "done"}
```

前端依次处理：`token` 追加渲染 → `sources` 渲染引用面板 → `done` 结束加载态。

### 健康检查

`GET /api/health` → `{"status": "ok"}`，被 docker-compose healthcheck 和部署脚本使用。

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

---

## License

MIT
