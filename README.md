# 中华药典智能问答系统

基于《中国药典》2020 年版一部构建的端到端 RAG（检索增强生成）系统，覆盖从原始文档解析到智能问答的完整流水线。

系统从 54,442 段原始 Word 文档中提取 **2,300+ 个药品条目**、生成 **11,369 个语义切片**，通过向量检索 + BM25 + RRF 融合 + 元数据过滤的混合检索架构，实现 **Hit@5 = 94%** 的检索准确率（[LangChain 标准版](docs/07_LangChain标准版架构.md) Ensemble 融合最优配置），并集成大模型（LongCat / 任意 OpenAI 兼容端点）提供专业问答服务。项目包含**手写实现**与 **LangChain 标准版**两套完全独立引擎，可并行运行、对比评估。

---

## 系统架构

```
中国药典 2020 一部.docx
        │
        ▼
┌───────────────────────────────────────────────┐
│  ETL 流水线                                     │
│  parser.py → cleaner.py → chunker.py           │
│  54,442 段落 → 2,387 药品 → 11,369 切片         │
└─────────────────────┬─────────────────────────┘
                      ▼
┌───────────────────────────────────────────────┐
│  索引构建                                       │
│  FAISS 向量索引 + BM25 关键词索引 + SQLite 元数据│
└─────────────────────┬─────────────────────────┘
                      ▼
┌───────────────────────────────────────────────┐
│  检索引擎                                       │
│  查询解析 → 向量+BM25 → RRF融合 → 重排 → 上下文  │
└─────────────────────┬─────────────────────────┘
                      ▼
┌───────────────────────────────────────────────┐
│  生成引擎                                       │
│  指代消解 → 检索 → Prompt构建 → LLM生成 → 后处理 │
└─────────────────────┬─────────────────────────┘
                      ▼
┌──────────────────┬────────────────────────────┐
│  FastAPI 后端     │  Streamlit Web UI          │
│  RESTful API     │  5 页面可视化交互            │
│  SSE 流式输出     │  流式对话 + 引用展示         │
└──────────────────┴────────────────────────────┘
```

---

## 项目结构

```
Chinese-Medicine/
├── data/
│   ├── raw/                      # 原始 Word 文档
│   │   └── 2020年药典一部.docx
│   ├── processed/                # ETL 输出
│   │   ├── drugs.json            # 2,387 个结构化药品条目
│   │   └── chunks.json           # 11,369 个语义切片
│   ├── vectorstore/              # 索引文件
│   │   ├── chroma/               # FAISS 向量索引 (46.6 MB)
│   │   ├── bm25_index.pkl        # BM25 关键词索引 (24.5 MB)
│   │   └── metadata.db           # SQLite 元数据库 (16.5 MB)
│   └── eval/
│       ├── test_queries.json     # 100 题测试集
│       └── reports/              # 评估报告
├── models/
│   └── bge-large-zh-v1.5/        # 本地 Embedding 模型
├── src/
│   ├── config.py                 # 全局配置
│   ├── etl/                      # 数据处理流水线
│   │   ├── parser.py             # Word 文档解析器
│   │   ├── cleaner.py            # 数据清洗（OCR纠错 + 去空格）
│   │   └── chunker.py            # 结构化语义切片
│   ├── indexing/                  # 索引构建
│   │   ├── embedder.py           # BGE-large-zh 向量编码
│   │   ├── vector_index.py       # FAISS 向量索引
│   │   ├── keyword_index.py      # BM25 关键词索引
│   │   ├── metadata_store.py     # SQLite 元数据存储
│   │   └── fusion.py             # RRF 多路融合
│   ├── retrieval/                # 检索引擎
│   │   ├── query_parser.py       # 查询解析（药品名/章节/意图）
│   │   ├── reranker.py           # BGE-Reranker-v2-m3 重排
│   │   └── retriever.py          # 主检索引擎
│   ├── generation/               # 生成引擎
│   │   ├── prompts.py            # Prompt 模板
│   │   ├── llm_client.py         # LongCat 2.0 LLM 客户端
│   │   ├── dialogue.py           # 多轮对话管理（指代消解）
│   │   ├── postprocessor.py      # 回答后处理（引用/校验/美化）
│   │   └── generator.py          # 主生成引擎
│   ├── api/                      # API 服务
│   │   ├── main.py               # FastAPI 应用
│   │   ├── schemas.py            # 请求/响应模型
│   │   └── session.py            # 会话管理
│   └── webui/                    # Web UI
│       ├── app.py                # Streamlit 主应用
│       ├── api_client.py         # API 客户端
│       ├── components.py         # UI 组件
│       └── styles.py             # CSS 主题
├── langchain_app/                # LangChain 标准版（完全独立实现，零依赖 src/）
│   ├── build_index.py            # FAISS + BM25Retriever + drug_names 构建
│   ├── query_understanding.py    # 规则式查询理解（药品名扩展/横向分类）
│   ├── retrievers.py             # FAISS+BM25 → EnsembleRetriever(RRF) → CrossEncoder 重排
│   ├── chains.py                 # ★ 主程序：LCEL 声明式会话 RAG 链
│   ├── guard.py / prompts.py / postprocess.py  # 守卫/提示词/引用标注
│   ├── service.py / api.py       # 服务层 + FastAPI（8001，Schema 与手撕版一致）
│   ├── eval.py                   # 独立评估器（判分公式与主项目一致）
│   ├── tests/                    # 30 个单元测试（查询理解/守卫/后处理/判分/检索器）
│   └── _archive_custom/          # 旧自定义适配版存档（不参与运行）
├── scripts/
│   ├── run/                      # 启动脚本
│   │   ├── run_etl.py            # 运行 ETL 流水线
│   │   ├── run_api.py            # 启动 API 服务
│   │   ├── run_webui.py          # 启动 Web UI
│   │   ├── run_all.py            # 一键启动前后端
│   │   └── run_eval.py           # 运行评估
│   ├── build/
│   │   └── build_index.py        # 构建索引
│   ├── test/                     # 功能测试
│   │   ├── test_retrieval.py     # 检索引擎测试
│   │   ├── test_generation.py    # 生成引擎测试
│   │   ├── test_api.py           # API 端到端测试
│   │   ├── test_webui_api.py     # Web UI 客户端测试
│   │   └── test_styles.py        # UI 样式验证
│   ├── debug/                    # 调试与失败分析
│   └── check/                    # 数据检查与验证
├── docs/                         # 技术文档
│   ├── 01_检索优化历程.md         # Hit@5 从 46% 到 91% 的完整历程
│   ├── 02_ETL解析修复.md          # 样式覆盖 + OCR 纠错
│   ├── 03_检索策略优化.md         # 多药品过滤 + BM25 后过滤
│   └── 04_最终评估报告.md         # 最终评估结果
├── requirements.txt
├── requirements-langchain.txt    # LangChain 版依赖（与主依赖共存）
└── README.md
```

---

## 核心技术

### 1. ETL 流水线 — 结构化语义切分

药典文档有天然的语义边界（【性状】【鉴别】【检查】等章节标记），简单按字符数切分会导致语义割裂。系统采用**三级结构化语义切分**策略：

| 层级 | 策略 | 说明 |
|------|------|------|
| Level 1 | 药品边界 | 每个药品/子剂型/饮片为独立单位 |
| Level 2 | 章节边界 | 按【性状】【鉴别】等章节标记分割 |
| Level 3 | 智能分组 | 短章节合并、长章节按成分切分、超长章节滑动窗口 |

**Chunk 类型**：

| 类型 | 说明 | 示例 |
|------|------|------|
| `overview` | 药品概述 | 来源、植物描述 |
| `clinical` | 临床应用 | 性味归经 + 功能主治 + 用法用量 + 注意 + 贮藏 |
| `assay` | 含量测定 | 按成分二次切分（如黄苓、金银花、连翘分别描述） |
| `detailed` | 详细信息 | 性状、鉴别、检查等中等长度章节 |
| `table` | 表格数据 | 检测方法中的规格表等 |

**OCR 纠错**：文档中 芪/芩/芎 等字因 OCR 识别错误完全缺失，通过 `DRUG_NAME_OCR_FIXES` 映射表在清洗阶段自动纠正（如 黄茂→黄芪、川弯→川芎）。

### 2. 混合检索架构

```
用户查询 → QueryParser 提取药品名/章节/意图
              │
       ┌──────┴──────┐
       ▼             ▼
  向量检索        BM25检索
  (FAISS)        (jieba+BM25Okapi)
  top_k=15       top_k=15(过滤时×4)
  +元数据过滤     +SQLite后过滤
       │             │
       └──────┬──────┘
              ▼
        RRF 融合 (k=60)
        top_n=30 候选
              │
              ▼
     BGE-Reranker-v2-m3 重排
        top_k=5 最终结果
```

**关键技术点**：

- **查询解析**：从用户查询中识别药品名、章节名、查询意图，生成元数据过滤条件
- **过滤器自动扩展**：将"人参"自动扩展为 ["人参", "人参-饮片", "人参叶"] 等变体
- **多药品过滤**：对跨品种比较查询（如"丹参和川芎的活血作用比较"）使用 `$in` 算子同步过滤向量检索和 BM25
- **BM25 后过滤**：BM25 结果通过 SQLite 元数据进行双向子串匹配后过滤

### 3. 生成引擎

| 模块 | 功能 |
|------|------|
| **DialogueManager** | 多轮对话管理，通过 LLM 将"它"→"人参"进行指代消解 |
| **Prompt 模板** | System Prompt 限定只基于药典原文回答，不编造信息 |
| **LLMClient** | 调用美团 LongCat 2.0（OpenAI 兼容接口） |
| **PostProcessor** | 引用标注 + 一致性校验（防幻觉）+ 格式美化 + 用药安全提醒 |

### 4. API 与 Web UI

**API 端点**：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/chat` | POST | 智能问答（支持多轮对话） |
| `/api/v1/chat/stream` | POST | 流式问答（SSE 实时输出） |
| `/api/v1/search` | GET | 独立检索查询 |
| `/api/v1/drugs` | GET | 药品列表查询 |
| `/api/v1/stats` | GET | 系统统计信息 |
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/sessions` | GET/POST/DELETE | 会话管理 |
| `/docs` | GET | Swagger UI 文档 |

**Web UI 页面**：智能问答（流式对话）、检索查询、药品浏览、系统统计、关于系统

---

## 评估结果

### 总体指标

| 指标 | 结果 | 目标 | 状态 |
|------|------|------|------|
| **Hit@1** | 91.00% | ≥70% | ✅ |
| **Hit@3** | 91.00% | ≥80% | ✅ |
| **Hit@5** | 91.00% | ≥90% | ✅ |
| **MRR** | 0.9100 | ≥0.80 | ✅ |
| **P50 延迟** | 0.834s | - | - |
| **P95 延迟** | 1.495s | - | - |

> 注：上表为 2026-07-06 报告数据。2026-08-30 复测与消融实验见 [05_LangChain对比实验](docs/05_LangChain对比实验.md) 与 [07_LangChain标准版架构](docs/07_LangChain标准版架构.md)——当前最优配置为 LangChain 标准版 Ensemble 融合（未启用重排）：Hit@5 = 94%，MRR = 0.9166，P50 = 0.051s。

### 分类型结果

| 查询类型 | 题数 | Hit@5 | MRR | 状态 |
|---------|------|-------|-----|------|
| 精确数值查询 | 15 | 100% | 1.0000 | ✅ |
| 跨品种比较查询 | 10 | 100% | 1.0000 | ✅ |
| 方法通则查询 | 10 | 100% | 1.0000 | ✅ |
| 多轮对话 | 10 | 100% | 1.0000 | ✅ |
| 否定性问题 | 5 | 100% | 1.0000 | ✅ |
| 单药品单属性查询 | 25 | 96% | 0.9600 | ✅ |
| 单药品多属性查询 | 15 | 93.33% | 0.9333 | ✅ |
| 横向条件查询 | 10 | 30% | 0.3000 | ⚠️ |

### 优化历程

| 阶段 | Hit@5 | 核心改动 |
|------|-------|---------|
| 初始版本 | 46% | 基础 ETL + 向量检索 + BM25 + RRF |
| 启发式边界检测 | 55% | 扩展段落样式识别 + 启发式药品名校验 |
| 数据规范化 | 64% | 药品名去空格 + BM25 过滤 |
| BM25 过滤修复 | 76% | BM25 检索增加 SQLite 元数据后过滤 |
| **ETL+OCR+多药品修复** | **91%** | 样式覆盖扩展 + OCR 纠错 + 多药品过滤 |

---

## 快速开始

### 环境准备

```bash
# 创建并激活 Conda 环境
conda create -n llm python=3.10
conda activate llm

# 安装依赖
pip install -r requirements.txt

# 设置 LongCat API Key（生成引擎需要）
# PowerShell:
$env:LONGCAT_API_KEY="your_api_key_here"
# CMD:
set LONGCAT_API_KEY=your_api_key_here
```

获取 API Key：访问 [LongCat API 开放平台](https://longcat.chat/platform/api_keys)

### 一键运行

```bash
# 1. 运行 ETL 流水线（解析 → 清洗 → 切片）
python scripts/run/run_etl.py

# 2. 构建索引（向量 + BM25 + SQLite）
python scripts/build/build_index.py

# 3. 一键启动前后端服务
python scripts/run/run_all.py
```

启动后访问：
- **Web UI**：http://localhost:8501
- **API 文档**：http://localhost:8000/docs

### 分步运行

```bash
# 启动 API 后端
python scripts/run/run_api.py [--host 0.0.0.0] [--port 8000] [--reload]

# 启动 Web UI（需先启动 API）
python scripts/run/run_webui.py

# 运行检索评估
python scripts/run/run_eval.py --mode retrieval

# 运行生成评估
python scripts/run/run_eval.py --mode generation

# 测试检索引擎
python scripts/test/test_retrieval.py

# 测试生成引擎
python scripts/test/test_generation.py

# 测试 API 端点
python scripts/test/test_api.py
```

### 构建索引选项

```bash
# 完整构建（含向量编码，需 GPU 约 5 分钟）
python scripts/build/build_index.py

# 快速构建（跳过向量编码，仅 BM25 + SQLite，约 10 秒）
python scripts/build/build_index.py --skip-embedding

# 构建后运行混合检索测试
python scripts/build/build_index.py --test
```

### LangChain 标准版（第二套引擎）

```bash
pip install -r requirements-langchain.txt

# 构建标准版索引（FAISS 已有则复用，仅 BM25 + 药品名清单约 11 秒）
python langchain_app/build_index.py

# 启动标准版 API（端口 8001，与手撕版 8000 可并行）
python scripts/run/run_lc_api.py
# 前端零改动切换引擎：API_BASE_URL=http://127.0.0.1:8001 streamlit run src/webui/app.py

# 评估与消融（不需要 LLM 额度）
python scripts/run/run_eval_lc.py [--no-rerank] [--no-bm25]
python scripts/run/run_eval_lc.py --mode generation   # 生成评估（需 LLM 额度）

# 单元测试（30 个）
python -m pytest langchain_app/tests/ -v
```

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| Embedding | BAAI/bge-large-zh-v1.5 | 1024 维中文语义向量 |
| 向量检索 | FAISS | IndexFlatIP 精确内积检索 |
| 关键词检索 | rank-bm25 + jieba | 中文分词 + BM25Okapi |
| 重排模型 | BAAI/bge-reranker-v2-m3 | CrossEncoder 跨语言重排（消融验证为负收益，默认关闭） |
| LLM | 可配置 OpenAI 兼容端点 | LongCat 2.0 / Agnes 等，.env 一行切换（见 docs/06） |
| LangChain 引擎 | LangChain 1.x 标准组件 | EnsembleRetriever(RRF) / LCEL / RunnableWithMessageHistory（独立实现） |
| API 框架 | FastAPI + Uvicorn | 异步高性能 + 自动文档 |
| Web UI | Streamlit | 数据应用可视化 |
| 元数据 | SQLite | 结构化过滤与查询 |
| 文档解析 | python-docx | Word 文档结构化提取 |

---

## 数据规模

| 指标 | 数值 |
|------|------|
| 原始文档段落 | 54,442 |
| 药品条目数 | 2,387 |
| 语义切片数 | 11,369 |
| 药品名数（去重） | 2,378 |
| FAISS 索引 | 46.6 MB |
| BM25 索引 | 24.5 MB |
| SQLite 元数据库 | 16.5 MB |
| 测试集题数 | 100 |

---

## 配置说明

核心配置集中在 `src/config.py`，主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VECTOR_TOP_K` | 15 | 向量检索召回数 |
| `BM25_TOP_K` | 15 | BM25 检索召回数（过滤时 ×4） |
| `RRF_K` | 60 | RRF 融合平滑参数 |
| `RRF_TOP_N` | 30 | RRF 融合后保留候选数 |
| `RERANKER_TOP_K` | 5 | 重排后返回结果数 |
| `ENABLE_RERANKER` | True | 是否启用重排 |
| `LLM_TEMPERATURE` | 0.3 | LLM 生成温度 |
| `LLM_MAX_TOKENS` | 2048 | 单次回答最大 token |
| `DIALOGUE_MAX_HISTORY` | 5 | 多轮对话保留轮数 |
| `ENABLE_CITATION` | True | 引用标注 |
| `ENABLE_CONSISTENCY_CHECK` | True | 一致性校验（防幻觉） |

---

## 后续优化方向

### 检索与生成质量

1. **BM25 分词注入领域词典**：手撕版 `JiebaTokenizer` 注入了药品名等领域词（freq=10000），标准版尚未做——部分药品的临床条目召回弱于手撕版，优先级最高
2. **数值密集查询的生成约束**：一致性校验实测两版各检出 12%/13%，检出集中在含量测定类——可在该类查询时提高数值条目召回权重，或在 Prompt 中强制"数值逐字引用原文"
3. **药材别名映射**：查询"川芎"时自动扩展为含川芎的复方制剂（川芎茶调丸等）
4. **补充数据源**：川芎原药材、薏苡仁等在药典文档中缺失的条目
5. **重排器复验**：当前测试集（100 题）证明重排为负收益，在更大规模评测集上复查该结论；检索/生成两侧分别消融

### 评测体系升级

6. **语义匹配评估**：当前关键词/子串判分对改写与全角标点敏感（"6〜12g" vs "6~12g"），引入 embedding 相似度或 LLM-as-judge
7. **评测集扩容与分层**：100 题偏小，扩至 500+ 并保持 8 类型分层覆盖；评测进 CI，每次改动自动回归 Hit@5

### 工程化与规模化

8. **部署**：Docker 化两套引擎 + docker-compose 编排（API/WebUI/索引构建）
9. **LLM 服务升级**：当前生成 P95 受免费推理通道拖累（~23s），切换生产级端点
10. **规模化改造路径**（当前单机内存索引，11k 规模够用，百万级需）：FAISS Flat → Milvus/pgvector/HNSW；会话内存 dict → Redis；GPU 重排拆独立推理服务；检索命中率/引用率/延迟分位埋点

> 已完成：~~横向条件查询分类过滤~~（标准版修复分类映射缺陷，10%→70%，见 [docs/05](docs/05_LangChain对比实验.md) / [docs/07](docs/07_LangChain标准版架构.md)）

---

## 相关文档

- [检索优化历程](docs/01_检索优化历程.md) — Hit@5 从 46% 到 91% 的完整过程
- [ETL 解析修复](docs/02_ETL解析修复.md) — 样式覆盖扩展 + OCR 字符纠正
- [检索策略优化](docs/03_检索策略优化.md) — 多药品过滤 + BM25 后过滤 + 过滤器扩展
- [最终评估报告](docs/04_最终评估报告.md) — 100 题测试集详细评估结果
- [LangChain 对比实验](docs/05_LangChain对比实验.md) — 手撕版 vs LangChain vs 纯标准组件三方对比
- [LLM 提供商配置](docs/06_LLM提供商配置.md) — .env 切换 LLM 端点（LongCat / Agnes 等）
- [LangChain 标准版架构](docs/07_LangChain标准版架构.md) — 完全独立实现 + 消融实验

---

## 更新记录

### 2026-08-30 LangChain 标准版（完全独立实现）+ 消融实验

将 LangChain 引擎重写为**完全独立的标准组件实现**（不依赖 `src/` 任何模块，可单独交付），旧的自定义适配版归档至 `langchain_app/_archive_custom/`。架构与踩坑记录见 [docs/07](docs/07_LangChain标准版架构.md)。

**核心结果**（同 100 题测试集、同判分公式）：

| 引擎 | Hit@5 | MRR | P50 延迟 |
|------|-------|-----|---------|
| 手撕版（当前代码实测） | 89% | 0.8900 | 0.727s |
| LangChain 标准版（完整混合+重排） | 92% | 0.9120 | 0.683s |
| **LangChain 标准版（Ensemble 融合，未启用重排）** | **94%** | **0.9166** | **0.051s** |
| 标准版消融：纯向量+重排 | 92% | 0.9008 | 0.360s |

**关键发现**：CrossEncoder 重排在本数据集上为负收益（-2pp Hit@5、13 倍延迟），最优上线配置为 Ensemble 融合不加重排——用测量代替假设。

**生成侧对照实验**（同 100 题、同通道 agnes-2.5-flash）：标准版关键词覆盖率 60.5%（手撕版 58.6%）、引用率 100%（92%）、延迟持平；生成侧重排同样无收益（59.2% vs 60.5%），"去重排"在检索与生成两侧均成立。**数值一致性校验已移植到标准版**（与手撕版同口径同逻辑），实测检出率 13% vs 手撕版 12%，同量级。详见 [docs/07](docs/07_LangChain标准版架构.md)。

**标准组件栈**：

| 环节 | 组件 |
|---|---|
| 检索融合 | `FAISS` + `BM25Retriever` → `EnsembleRetriever`（内置 RRF） |
| 重排（可选） | `HuggingFaceCrossEncoder` + `BaseDocumentCompressor` |
| 主链 | LCEL 声明式 + `RunnableBranch` 守卫短路 + `RunnableWithMessageHistory` 会话记忆 |
| 提示词 | `ChatPromptTemplate` + `MessagesPlaceholder`（多轮 condense-question 改写） |

**运行方式**：

```bash
pip install -r requirements-langchain.txt
python langchain_app/build_index.py          # 构建索引（FAISS 已有则复用）
python scripts/run/run_lc_api.py             # API 服务（8001，前端零改动切换）
python scripts/run/run_eval_lc.py [--no-rerank] [--no-bm25]   # 评估与消融
```

### 2026-08-30 问题领域守卫与 LLM 提供商

- LLM 接入切换为环境变量配置（`.env`），支持任意 OpenAI 兼容端点（当前 agnes-2.5-flash 免费通道，详见 [docs/06](docs/06_LLM提供商配置.md)）
- 兼容推理类模型（guard/dialogue 小预算调用适配 `reasoning_content`）

### 2026-07-07 问题领域守卫：非医药问题拦截

**更新内容：**

新增问题领域守卫模块（`src/generation/guard.py`），在 RAG 流程入口处检测用户问题是否与中医药/药典相关，对无关问题直接礼貌拒绝，不浪费检索和生成资源。

**检测策略（两层过滤）：**

1. **关键词快速通道（0ms）**：命中医药关键词（药、药材、性味、归经、人参、黄芪等 100+ 个）直接通过；命中明显无关关键词（股票、游戏、编程、足球等 60+ 个）直接拒绝
2. **LLM 语义判断（~0.5s）**：对关键词无法判定的模糊问题，用一次快速 LLM 调用（`temperature=0, max_tokens=10`）进行分类

**集成位置：**

在 `Generator.answer()` 和 `Generator.answer_stream()` 方法中，**检索之前**执行守卫检查。检测到非医药问题时直接返回拒绝回答，不执行检索和 LLM 生成。

**拒绝回答示例：**

> 抱歉，我是《中国药典》智能问答系统，只能回答与中药、药典相关的问题。
>
> 我可以帮您解答以下类型的问题：
> - 药材的性味归经、功能主治、用法用量
> - 药材的性状、鉴别、含量测定
> - 药材的炮制方法、贮藏条件
> - 药材间的功效对比
> - 药典通则方法（如色谱法、水分测定等）
>
> 请尝试提出与中医药相关的问题，例如："人参的性味归经是什么？"

**新增文件：**

| 文件 | 说明 |
|------|------|
| `src/generation/guard.py` | 问题领域守卫模块（关键词过滤 + LLM 语义判断） |

**修改文件：**

| 文件 | 改动 |
|------|------|
| `src/generation/generator.py` | 在 `answer()` 和 `answer_stream()` 中集成守卫检查 |

### 2026-07-07 系统优化：流式元数据 + 横向查询 + 守卫加速 + API 测试 + 配置外部化

本次更新涵盖五项优化，全面提升系统功能和可维护性。

---

#### 1. 流式问答返回来源与引用

**问题：** 流式接口（`/api/v1/chat/stream`）此前只返回文本片段和结束信号，前端无法获取检索来源、引用标注和一致性检查结果，导致流式模式下用户体验不完整。

**改动：**

- 重写 `Generator.answer_stream()`：从只 yield 文本片段改为 yield 事件字典
  - `{"content": "..."}` — 文本片段（多次）
  - `{"metadata": {...}}` — 结束时一次性返回 `sources`、`citations`、`consistency_issues`、`latency_ms` 等
  - `{"rejected": True, "content": "..."}` — 非医药问题拒绝
- 流式结束后自动执行后处理（引用标注 + 一致性校验）和对话历史更新，与同步接口行为一致
- 更新 API 端点 `chat_stream`：SSE 结束事件附带完整元数据
- 更新前端 `app.py`：流式结束后展示来源、引用、一致性警告

| 文件 | 改动 |
|------|------|
| `src/generation/generator.py` | `answer_stream()` 重写为事件生成器，结束后返回元数据 |
| `src/api/main.py` | `chat_stream` 端点解析事件并附带 sources/citations |
| `src/webui/app.py` | 流式模式展示来源、引用、一致性警告 |

---

#### 2. 横向条件查询优化

**问题：** 用户问「哪些药材有补气功效」「含挥发油的中药有哪些」时，系统进行全库检索，结果混入大量成方制剂条目，导致召回不精准。

**改动：**

- 在 `QueryParser` 中新增横向条件查询检测：
  - 识别「哪些药材」「什么中药」「列举」「有X功效」等句式
  - 检测到横向查询时，自动推断 `category_filter`（`药材` / `成方制剂`）
- 在 `Retriever` 中应用 category 过滤：
  - 向量检索：Chroma `where` 条件增加 `category` 字段
  - BM25 检索：结果后过滤，只保留匹配分类的条目

| 文件 | 改动 |
|------|------|
| `src/retrieval/query_parser.py` | 新增 `HORIZONTAL_KEYWORDS`、`_is_horizontal_query()`、`_detect_category_filter()` |
| `src/retrieval/retriever.py` | 向量检索和 BM25 检索增加 category 过滤逻辑 |

---

#### 3. 守卫模块多轮对话加速

**问题：** 多轮对话中每一轮都调用 LLM 做领域判断，增加 ~0.5s 延迟，但同一会话内用户通常会持续问医药相关问题。

**改动：**

- `GuardChecker` 新增 `_session_trusted` 状态：
  - 第一轮通过守卫后标记为信任
  - 后续轮次（`is_followup=True`）跳过 LLM 判断，直接通过关键词快通道
  - 新会话时调用 `reset_session()` 重置信任状态
- `Generator` 在调用 `guard.check()` 时传入 `is_followup` 参数
- API 端点在新会话时自动重置守卫信任状态

| 文件 | 改动 |
|------|------|
| `src/generation/guard.py` | `check()` 增加 `is_followup` 参数，新增 `reset_session()` |
| `src/generation/generator.py` | `answer()` 和 `answer_stream()` 传入 `is_followup` |
| `src/api/main.py` | 新会话时调用 `guard.reset_session()` |

---

#### 4. API 接口自动化测试

**问题：** 缺少系统化的 API 端点测试，每次修改后需要手动验证。

**改动：**

- 新建 `src/api/tests/test_endpoints.py`，使用 pytest 框架覆盖关键端点：
  - `TestHealth` — 健康检查 & 统计
  - `TestSearch` — 检索查询（基本检索、药品过滤、空查询）
  - `TestChat` — 智能问答（单轮、多轮对话、守卫拦截）
  - `TestChatStream` — 流式问答（基本流式、守卫拦截）
  - `TestSessions` — 会话管理（创建、删除、列表）
  - `TestDrugs` — 药品列表（列表、搜索）
- 同时支持 `pytest` 运行和 `python` 直接运行

**运行方式：**

```bash
# 方式一：pytest
pytest src/api/tests/test_endpoints.py -v

# 方式二：直接运行
python src/api/tests/test_endpoints.py
```

| 文件 | 说明 |
|------|------|
| `src/api/tests/test_endpoints.py` | API 端点 pytest 测试套件 |
| `src/api/tests/__init__.py` | 测试包初始化 |

---

#### 5. Reranker 模型路径外部化

**问题：** `RERANKER_MODEL_PATH` 硬编码为绝对路径，换机器部署时需要修改源码。

**改动：**

- `config.py` 和 `config.example.py` 中 `RERANKER_MODEL_PATH` 改为优先从环境变量 `RERANKER_MODEL_PATH` 读取

```python
RERANKER_MODEL_PATH = os.environ.get(
    "RERANKER_MODEL_PATH",
    r"D:\MODEL\BAAI\bge-reranker-v2-m3",  # 默认路径
)
```

| 文件 | 改动 |
|------|------|
| `src/config.py` | `RERANKER_MODEL_PATH` 改为环境变量优先 |
| `src/config.example.py` | 同步更新模板 |
