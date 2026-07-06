# 中华药典智能问答系统

基于《中国药典》2020 年版一部构建的端到端 RAG（检索增强生成）系统，覆盖从原始文档解析到智能问答的完整流水线。

系统从 54,442 段原始 Word 文档中提取 **2,389 个药品条目**、生成 **11,369 个语义切片**，通过向量检索 + BM25 + RRF 融合 + 重排的混合检索架构，实现 **Hit@5 = 91%** 的检索准确率，并集成美团 LongCat 2.0 大模型提供专业问答服务。

---

## 系统架构

```
中国药典 2020 一部.docx
        │
        ▼
┌───────────────────────────────────────────────┐
│  ETL 流水线                                     │
│  parser.py → cleaner.py → chunker.py           │
│  54,442 段落 → 2,389 药品 → 11,369 切片         │
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
│   │   ├── drugs.json            # 2,389 个结构化药品条目
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

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| Embedding | BAAI/bge-large-zh-v1.5 | 1024 维中文语义向量 |
| 向量检索 | FAISS | IndexFlatIP 精确内积检索 |
| 关键词检索 | rank-bm25 + jieba | 中文分词 + BM25Okapi |
| 重排模型 | BAAI/bge-reranker-v2-m3 | CrossEncoder 跨语言重排 |
| LLM | 美团 LongCat 2.0 | OpenAI 兼容接口 |
| API 框架 | FastAPI + Uvicorn | 异步高性能 + 自动文档 |
| Web UI | Streamlit | 数据应用可视化 |
| 元数据 | SQLite | 结构化过滤与查询 |
| 文档解析 | python-docx | Word 文档结构化提取 |

---

## 数据规模

| 指标 | 数值 |
|------|------|
| 原始文档段落 | 54,442 |
| 药品条目数 | 2,389 |
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

1. **横向条件查询**（当前 Hit@5 = 30%）：对"哪些药材具有补气功效"类查询，在检索时限定 `category = "药材和饮片"`，避免返回复方制剂
2. **药材别名映射**：查询"川芎"时自动扩展为含川芎的复方制剂（川芎茶调丸等）
3. **补充数据源**：川芎原药材、薏苡仁等在药典文档中缺失的条目

---

## 相关文档

- [检索优化历程](docs/01_检索优化历程.md) — Hit@5 从 46% 到 91% 的完整过程
- [ETL 解析修复](docs/02_ETL解析修复.md) — 样式覆盖扩展 + OCR 字符纠正
- [检索策略优化](docs/03_检索策略优化.md) — 多药品过滤 + BM25 后过滤 + 过滤器扩展
- [最终评估报告](docs/04_最终评估报告.md) — 100 题测试集详细评估结果
