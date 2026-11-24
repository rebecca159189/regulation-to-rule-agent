# 监管规则助手（Python · LangGraph Agent + 混合检索 RAG）

读一份监管通知 → 检索条款和平台字段 → 大模型通过**工具调用**写出结构化规则草稿 → **程序**逐项校验 → 人工复核 → 导出禁用状态的 `trans_rule` 配置。

[English](README.md)

本项目是个人原型：结合在银行交易反欺诈平台的工作经验，把“监管通知 → 风控规则配置”中反复核对字段、单位、名单版本的工作交给“模型理解 + 程序校验”。仓库里的平台字段、名单和绑定 ID 都是通用的演示数据。

---

## 一、快速开始（Windows）

**1. 安装 Python 3.12**：python.org 下载，安装时勾选 “Add python.exe to PATH”。不建议用 3.14 这类最新版本，部分依赖可能还没有对应的安装包。

**2. 安装依赖**（在 PowerShell 里，第一次要几分钟）：

```powershell
cd anti-fraud-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- 如果 `Activate.ps1` 报“禁止运行脚本”，先执行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，输入 Y 回车。
- 国内下载慢，可以在命令后面加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。
- 激活成功后，命令行最前面会出现 `(.venv)`。以后每次新开窗口，都要先 `cd` 到项目再执行 `.\.venv\Scripts\Activate.ps1`。

**3. 准备配置**：

```powershell
copy .env.example .env
notepad .env
```

不填 Key 也能启动，此时只能用“演示模式”。

**4. 启动**：

```powershell
streamlit run app.py
```

浏览器会自动打开 http://localhost:8501 。停止服务在窗口里按 Ctrl+C。

---

## 二、配置 DeepSeek（`.env`）

| 配置项 | 填什么 |
|---|---|
| `LLM_BASE_URL` | `https://api.deepseek.com`（已填好） |
| `LLM_API_KEY` | 你的 Key，`sk-` 开头 |
| `LLM_MODEL` | `deepseek-flash`（已填好） |
| `LLM_EXTRA_BODY` | 默认关闭 DeepSeek 的思考模式，**不要删**（原因见 `.env.example` 里的注释） |

改完 `.env` 要**重启**（Ctrl+C 后重新执行 `streamlit run app.py`）。左侧栏显示“大模型：deepseek-flash”就是配置成功了。

**向量检索**默认使用本机嵌入模型 `BAAI/bge-small-zh-v1.5`，第一次启动时会自动下载（约 90MB）。下载失败时会自动退回 BM25 关键词检索，左侧栏会显示原因。国内下载不了，就在 `.env` 里打开 `HF_ENDPOINT=https://hf-mirror.com` 这一行。

---

## 三、怎么体验

1. **演示模式（不花钱、不需要 Key）**：左侧选“演示模式”→ 新建任务 → 学习示例选“01 大于3000元”→ 载入 → ③ Agent 运行 → 从头运行 → ④ 复核与导出 → 填复核人和说明、勾选确认 → 批准并导出。
   演示模式用的是“剧本模型”，提交的是人工写好的标准答案，**不是大模型的结果**，只用来走一遍流程。
2. **真实模式**：配好 Key 后选“真实大模型”，用同样的步骤操作。在 ③ 能看到模型每次调用了什么工具、用了多少 tokens，以及它实际看到的完整对话。
3. **故意制造错误**：在 ④ 把比较符从 `gt` 改成 `gte`，点“保存修改并重新校验”，会看到 `OPERATOR_SOURCE` 和 `BOUNDARY` 两个错误；把单位 `yuan` 改成 `fen`，会看到 `MONEY_SOURCE` 错误。
4. **检索调试**：在 ② 输入关键词，能看到每个片段在 BM25 和向量检索里各排第几，以及 RRF 融合后的分数。
5. **试几个“陷阱”示例**：08 单位缺失、10 名单版本缺失，正确的表现是模型向你提问；09 累计窗口，正确的表现是标成“暂不能落实”。
6. **批量评测**：

```powershell
python scripts\eval.py --mode demo    # 只验证流程，应为 10/10
python scripts\eval.py --mode llm     # 真实模型跑 10 个示例，统计做对几个（会花少量 token）
```

7. **单元测试**：`python -m pytest -q`（45 个测试，不联网、不花钱）。

想清空所有数据，关掉服务后删除 `.data` 文件夹即可。

---

## 四、整体架构

```
                 ┌──────────── 校验不过 / 检索完，继续 ─────────────┐
                 ▼                                                  │
 START → prepare → agent ──(模型要调用工具)──► tools ───────────────┤
                    │                           ├─ ask_human ──► clarify ⏸ 等人回答 ─► agent
                    │ 调用次数用完               │
                    ▼                           └─ 校验通过 / 交给人工 ─► review ⏸ 等人复核
                   END                                                  ├─ 编辑 → 重新校验
                                                                        ├─ 退回给模型修改 → agent
                                                                        ├─ 驳回 → END
                                                                        └─ 批准 → export → END
```

- **prepare**：在“本次文件 / 平台资料 / 已有规则”三个范围各检索一次，拼成第一条消息。
- **agent**：调用大模型（绑定了 6 个工具）；每调用一次计数一次，超过调用上限就停。
- **tools**：执行模型申请的工具：检索、提问、提交草稿、转交人工。
- **submit_draft**：提交后立刻跑确定性校验，把具体错误返回给模型，让它自己修正。
- **clarify / review**：用 `interrupt()` 暂停整张图，状态存进 SQLite；你在页面上点按钮后再继续。

---

## 五、用到的技术（对照招聘里常见的关键词）

| 关键词 | 在本项目里的位置 | 说明 |
|---|---|---|
| **LangGraph / StateGraph** | `regrule/agent/graph.py` | 用“状态 + 节点 + 条件边”编排 Agent 流程 |
| **Tool Calling / Function Calling** | `regrule/agent/tools.py` | 模型只能“申请调用”这 6 个函数，真正执行的是我们的代码 |
| **ReAct 循环** | agent ⇄ tools | 思考 → 调工具 → 看结果 → 再思考 |
| **结构化输出 / Pydantic** | `regrule/schemas.py` | 草稿结构自动转成 JSON Schema 发给模型，返回后再校验 |
| **Human-in-the-loop** | `interrupt()`，在 clarify、review 节点 | 模型提问、人工复核时暂停等人 |
| **Checkpointer / 持久化** | `SqliteSaver`（`regrule/service.py`） | 每一步状态都存进 SQLite，关掉页面再打开还能接着跑 |
| **RAG** | `regrule/documents.py`、`retrieval.py` | 加载 → 切分 → 建索引 → 检索 → 交给模型 |
| **文本切分** | `RecursiveCharacterTextSplitter` | 800 字一片，相邻两片重叠 150 字 |
| **Embedding** | `regrule/embeddings.py` | bge-small-zh 本地模型，也可以换成接口 |
| **向量数据库** | Chroma（`.data/chroma`） | 存放向量，带元数据过滤（按范围隔离） |
| **混合检索 / BM25 / RRF** | `retrieval.py` | jieba 分词做 BM25，再和向量检索的结果用倒数排名融合 |
| **Guardrails / 护栏** | `regrule/validation.py` | 引文逐字核对、比较符词典、精确换算、边界测试，全部是普通代码 |
| **Prompt 工程** | `regrule/agent/prompts.py` | 系统提示词定角色和边界，上下文消息提供材料 |
| **评测（Eval）** | `scripts/eval.py` | 10 个带标准答案的样例，统计通过率 |
| **测试替身** | `demo_model.py`、`tests/` | 用假模型和假 HTTP 服务测试 Agent，不花钱 |

---

## 六、代码地图

```
app.py                     Streamlit 页面
regrule/
  config.py                读取 .env
  schemas.py               数据结构（Pydantic）
  platform.py              平台字段、名单、模板（演示映射）
  fixtures.py              10 个学习示例和标准答案
  store.py                 SQLite 存储
  documents.py             读文件、切条款、切片段
  embeddings.py            嵌入模型（向量化）
  retrieval.py             混合检索（BM25 + Chroma + RRF）
  validation.py            确定性校验 + 编译 trans_rule
  llm.py                   接入大模型（ChatOpenAI → DeepSeek）
  service.py               页面和脚本调用的统一入口
  agent/
    graph.py               LangGraph 工作流
    tools.py               6 个工具
    prompts.py             提示词
    demo_model.py          演示模式的剧本模型
scripts/eval.py            批量评测
tests/                     单元测试
.data/                     运行数据（自动生成）
```

建议的阅读顺序：`schemas.py` → `validation.py` → `documents.py` → `retrieval.py` → `agent/tools.py` → `agent/graph.py` → `app.py`。

---

## 七、目前没做的事

- 演示模式不是模型，10/10 只能说明流程能跑通。真实效果要配好 Key 后用 `eval.py --mode llm` 测。
- 没有“重排模型”（Rerank）和 OCR，扫描件读不了。
- 人工回答的问题**不能替代原文依据**：比如示例 08 原文没写单位，即使你回答“元”，校验也不会放行。这是有意为之，规则依据必须来自原文。
- 平台字段和绑定 ID 都是演示数据，还没在真实的规则引擎里加载验证过。
- 单人本地使用，没有登录和权限；也没有接入 LangSmith 这类调用追踪工具。

以后可以加：用 FastAPI 把它做成接口服务、Docker 部署、加重排模型、接入 LangSmith 做观测。
