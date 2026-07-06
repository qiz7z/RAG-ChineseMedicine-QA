# 中华药典 RAG 系统

基于《中国药典》2020年版一部的检索增强生成（Retrieval-Augmented Generation）系统。

本项目针对中医药典数据的特殊性，设计了**结构化语义切分**策略，确保检索单元语义完整、上下文连贯，为下游向量检索与 LLM 生成提供高质量知识库。

---

## 目录

- [项目背景](#项目背景)
- [项目结构](#项目结构)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [数据处理流水线（ETL）](#数据处理流水线etl)
  - [阶段一：文档解析（Parser）](#阶段一文档解析parser)
  - [阶段二：数据清洗（Cleaner）](#阶段二数据清洗cleaner)
  - [阶段三：智能切片（Chunker）](#阶段三智能切片chunker)
- [核心设计：数据切割策略详解](#核心设计数据切割策略详解)
  - [为什么不能简单按字符数切？](#为什么不能简单按字符数切)
  - [三级结构化语义切分策略](#三级结构化语义切分策略)
  - [Chunk 类型说明](#chunk-类型说明)
  - [Chunk 元数据说明](#chunk-元数据说明)
  - [切片参数配置](#切片参数配置)
- [输出数据格式](#输出数据格式)
- [索引构建（阶段二）](#索引构建阶段二)
  - [混合检索架构](#混合检索架构)
  - [向量索引（FAISS）](#向量索引faiss)
  - [关键词索引（BM25）](#关键词索引bm25)
  - [元数据存储（SQLite）](#元数据存储sqlite)
  - [RRF 多路融合](#rrf-多路融合)
  - [索引构建脚本](#索引构建脚本)
  - [索引配置参数](#索引配置参数)
  - [混合检索验证](#混合检索验证)
- [检索引擎（阶段三）](#检索引擎阶段三)
  - [检索流程总览](#检索流程总览)
  - [查询解析器（QueryParser）](#查询解析器queryparser)
  - [重排模型（Reranker）](#重排模型reranker)
  - [主检索引擎（Retriever）](#主检索引擎retriever)
  - [上下文组装](#上下文组装)
  - [检索引擎配置参数](#检索引擎配置参数)
  - [检索引擎使用示例](#检索引擎使用示例)
- [生成引擎（阶段四）](#生成引擎阶段四)
  - [生成流程总览](#生成流程总览)
  - [LLM 客户端（LLMClient）](#llm-客户端llmclient)
  - [Prompt 模板设计](#prompt-模板设计)
  - [回答后处理（PostProcessor）](#回答后处理postprocessor)
  - [多轮对话管理（DialogueManager）](#多轮对话管理dialoguemanager)
  - [主生成引擎（Generator）](#主生成引擎generator)
  - [生成引擎配置参数](#生成引擎配置参数)
  - [生成引擎使用示例](#生成引擎使用示例)
- [API 服务（阶段五）](#api-服务阶段五)
  - [API 端点总览](#api-端点总览)
  - [智能问答接口](#智能问答接口)
  - [流式问答接口](#流式问答接口)
  - [检索查询接口](#检索查询接口)
  - [药品列表接口](#药品列表接口)
  - [会话管理接口](#会话管理接口)
  - [系统接口](#系统接口)
  - [API 启动与配置](#api-启动与配置)
- [Web UI（阶段六）](#web-ui阶段六)
  - [技术架构](#技术架构)
  - [页面功能](#页面功能)
  - [UI 启动与配置](#ui-启动与配置)
- [后续阶段](#后续阶段)

---

## 项目背景

《中国药典》是药品生产、经营、使用和监督管理等均应遵循的法定技术规范。2020年版一部收载药材和饮片、植物油脂和提取物、成方制剂和单味制剂等，共计数千个药品条目。

本项目目标：构建一个基于 RAG 架构的药典智能问答系统，用户可以用自然语言提问（如"人参的性味归经是什么？""双黄连口服液的含量测定方法是什么？"），系统从药典知识库中检索相关内容并生成准确回答。

---

## 项目结构

```
中华药典项目/
├── data/
│   ├── raw/                          # 原始数据
│   │   └── 2020年药典一部.docx       # 药典原文（Word格式）
│   ├── processed/                    # 处理后的结构化数据
│   │   ├── drugs.json                # 清洗后的药品条目（结构化）
│   │   └── chunks.json               # 切片结果（RAG检索单元）
│   └── vectorstore/                  # 索引数据
│       ├── faiss/                    #   FAISS 向量索引 + 元数据
│       ├── bm25_index.pkl            #   BM25 关键词索引
│       └── metadata.db               #   SQLite 元数据库
├── src/
│   ├── config.py                     # 全局配置（路径、参数、OCR映射表等）
│   ├── etl/                          # ETL 流水线
│   │   ├── parser.py                 # 文档解析器
│   │   ├── cleaner.py                # 数据清洗器
│   │   └── chunker.py                # 智能切片器
│   ├── indexing/                     # 索引构建模块
│   │   ├── embedder.py               #   BGE Embedding 模型封装
│   │   ├── vector_index.py           #   FAISS 向量索引管理
│   │   ├── keyword_index.py          #   BM25 关键词索引（jieba 分词）
│   │   ├── metadata_store.py         #   SQLite 元数据存储
│   │   └── fusion.py                 #   RRF 多路检索融合
│   ├── retrieval/                    # 检索引擎模块
│   │   ├── query_parser.py           #   查询解析器（药品名/章节/意图识别）
│   │   ├── reranker.py               #   BGE-Reranker-v2-m3 重排模型封装
│   │   └── retriever.py              #   主检索引擎（多路召回→融合→重排→上下文）
│   ├── generation/                    # 生成引擎模块
│   │   ├── llm_client.py            #   LLM 客户端（LongCat 2.0 API 封装）
│   │   ├── prompts.py               #   Prompt 模板设计
│   │   ├── postprocessor.py         #   回答后处理（引用标注/一致性校验）
│   │   ├── dialogue.py              #   多轮对话管理（指代消解）
│   │   └── generator.py             #   生成引擎主模块（整合检索+生成）
│   └── api/                          # API 服务模块
│       ├── schemas.py               #   请求/响应数据模型（Pydantic）
│       ├── session.py               #   会话管理器（多轮对话 session）
│       └── main.py                  #   FastAPI 应用与路由定义
├── scripts/
│   ├── run_etl.py                    # ETL 一键运行脚本
│   ├── build_index.py                # 索引一键构建脚本
│   ├── test_retrieval.py             # 检索引擎测试脚本
│   ├── test_generation.py            # 生成引擎测试脚本
│   ├── test_api.py                   # API 端到端测试脚本
│   └── run_api.py                    # API 服务启动脚本
├── tests/                            # 测试目录
├── requirements.txt                  # Python 依赖
├── 项目书.md                          # 完整项目书
└── README.md                         # 本文件
```

---

## 环境准备

```bash
# 创建 conda 环境（推荐，含 PyTorch + CUDA）
conda create -n llm python=3.12
conda activate llm

# 安装 PyTorch（GPU 版，按实际 CUDA 版本调整）
# 本项目使用 PyTorch 2.10.0 + CUDA 12.8
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

# 安装项目依赖
pip install -r requirements.txt
```

当前依赖：

| 包 | 版本 | 用途 |
|---|---|---|
| `python-docx` | >=0.8.11 | 解析 .docx 文档 |
| `sentence-transformers` | >=2.2.0 | BGE Embedding 模型加载 |
| `faiss-cpu` | >=1.7.0 | 向量检索引擎（IndexFlatIP） |
| `rank-bm25` | >=0.2.2 | BM25 关键词检索 |
| `jieba` | >=0.42.1 | 中文分词（含药典自定义词典） |
| `numpy` | >=1.24.0 | 向量运算 |
| `torch` | >=2.0.0 | PyTorch（Embedding 模型推理） |
| `openai` | >=1.0.0 | OpenAI SDK（调用 LongCat 2.0 API） |
| `fastapi` | >=0.100.0 | Web API 框架（自动生成 OpenAPI 文档） |
| `uvicorn` | >=0.20.0 | ASGI 服务器（运行 FastAPI 应用） |

> **GPU 加速**：向量编码使用 BGE-large-zh 模型（1024 维），GPU 编码 11k chunk 约 5 分钟，CPU 约需 30 分钟以上。建议使用 CUDA 环境。

---

## 快速开始

### 1. 数据处理（ETL）

将药典 `.docx` 文件放入 `data/raw/` 目录后，执行：

```bash
conda activate llm
python scripts/run/run_etl.py
```

该命令会自动完成 **解析 → 清洗 → 切片** 三个阶段，输出：

- `data/processed/drugs.json` — 结构化药品条目
- `data/processed/chunks.json` — RAG 检索用文本块

### 2. 索引构建

ETL 完成后，构建三路索引（向量 + BM25 + SQLite）：

```bash
# 完整构建（含向量编码，需 GPU 约 5 分钟）
python scripts/build/build_index.py

# 快速构建（跳过向量编码，仅 BM25 + SQLite，约 10 秒）
python scripts/build/build_index.py --skip-embedding

# 构建后运行混合检索测试
python scripts/build/build_index.py --test
```

输出文件：

- `data/vectorstore/faiss/` — FAISS 向量索引 + pickle 元数据
- `data/vectorstore/bm25_index.pkl` — BM25 关键词索引
- `data/vectorstore/metadata.db` — SQLite 元数据库

### 3. 智能问答（生成引擎）

索引构建完成后，即可使用生成引擎进行智能问答：

```bash
# 设置 LongCat API Key（美团 LongCat 2.0）
# PowerShell:
$env:LONGCAT_API_KEY="your_api_key_here"
# CMD:
set LONGCAT_API_KEY=your_api_key_here

# 运行生成引擎测试
python scripts/test/test_generation.py
```

获取 API Key：访问 [LongCat API 开放平台](https://longcat.chat/platform/api_keys)

---

## 数据处理流水线（ETL）

ETL 流水线由三个阶段组成，每个阶段对数据进行逐步提炼：

```
2020年药典一部.docx
        │
        ▼
  ┌─────────────┐
  │  1. 解析器   │  parser.py — 提取药品条目、章节、表格
  │  Parser     │  .docx → DrugEntry 列表
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  2. 清洗器   │  cleaner.py — OCR纠错、全角半角统一、空白清理
  │  Cleaner    │  DrugEntry → 清洗后的 DrugEntry
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  3. 切片器   │  chunker.py — 结构化语义切分
  │  Chunker    │  DrugEntry → Chunk 列表
  └──────┬──────┘
         │
         ▼
   chunks.json（11,056 个检索单元）
```

### 阶段一：文档解析（Parser）

**文件**: `src/etl/parser.py`

从 `.docx` 文件中提取结构化的药品数据。利用 Word 文档的样式信息（Heading #3、Heading #4 等）识别药品层级结构。

**文档层级识别**：

```
药典一部
└── 药品条目 (Heading #3)              ← 主条目（药材/制剂名，含中文）
    ├── 拼音名 (Body text|3)
    ├── 拉丁名 (Body text|3)
    ├── 来源/概述 (正文)
    ├── 【章节1】 (正文，标记嵌在段首)    ← 如【性状】【鉴别】
    ├── 【章节2】
    │   └── 表格
    ├── 饮片 (Heading #4 / Body text|4) ← 饮片子节
    │   ├── 【炮制】
    │   └── 【性味与归经】...
    └── 子剂型 (Heading #4)             ← 如同一药品的不同剂型
        ├── 拼音名
        ├── 【处方】
        └── 【制法】...
```

**关键处理**：

- 使用 `iter_block_items` 保持段落与表格的原始文档顺序
- 将 docx 表格转换为 Markdown 格式，保留结构信息
- 根据 `【xxx】` / `〔xxx〕` 标记识别章节边界
- 自动推断药品分类（药材和饮片 / 成方制剂 / 植物油脂和提取物）

**输出**：`DrugEntry` 列表，每条目包含药品名、拼音名、拉丁名、概述文本和多个 `Section`（章节）。

### 阶段二：数据清洗（Cleaner）

**文件**: `src/etl/cleaner.py`

对解析后的数据进行多维度清洗，修复 OCR 识别错误和格式不一致问题。

**清洗步骤**：

| 步骤 | 说明 | 示例 |
|------|------|------|
| 1. 章节标记纠错 | 修复 OCR 导致的章节标记乱码 | `【検查】` → `【检查】`，`【含量測定】` → `【含量测定】` |
| 2. 章节名称归一化 | 统一章节名称的多种写法 | `功能主治` → `功能与主治`，`用法` → `用法与用量` |
| 3. 上下文敏感纠错 | 根据上下文修复字符级 OCR 错误 | `皂昔` → `皂苷`，`通则。512` → `通则0512` |
| 4. 全角→半角转换 | 统一数字和字母格式 | `０１２３` → `0123` |
| 5. 空白清理 | 合并多余空格、空行 | 去除连续空行、行首尾空格 |
| 6. 异常字符清理 | 去除 OCR 残留的特殊符号 | `■` → `。` |

**OCR 错误对照表（部分）**：

| 错误形式 | 正确形式 | 说明 |
|----------|----------|------|
| `検查` | `检查` | 日文汉字混入 |
| `含量測定` | `含量测定` | 繁体字混入 |
| `礎藏` / `疋藏` / `St藏` | `贮藏` | OCR 误识别 |
| `皂昔` | `皂苷` | 高频化学名词误识别 |
| `性昧与归经` | `性味与归经` | 字形相近误识别 |

### 阶段三：智能切片（Chunker）

**文件**: `src/etl/chunker.py`

这是 ETL 流水线的核心环节，将结构化药品数据切分为适合 RAG 检索的文本块（Chunk）。详见下方 [核心设计](#核心设计数据切割策略详解) 章节。

---

## 核心设计：数据切割策略详解

### 为什么不能简单按字符数切？

药典数据有天然的语义边界——`【性状】`、`【鉴别】`、`【检查】` 等章节标记。如果简单地按固定字符数切分（如每 500 字符切一段），会导致以下严重问题：

| 问题 | 说明 | 后果 |
|------|------|------|
| **语义截断** | `【性状】本品呈圆柱形...` 从中间被截断 | 检索到的 chunk 不完整，LLM 无法理解 |
| **章节混杂** | `【含量测定】黄苓...` 和 `【含量测定】金银花...` 混在同一个 chunk | 不同成分的测定方法混在一起，检索不精确 |
| **上下文丢失** | 药品名与正文被切到不同 chunk | 检索到的内容不知道属于哪个药品 |
| **临床信息割裂** | `【性味与归经】`、`【功能与主治】`、`【用法与用量】` 被分别切到不同 chunk | 用户问"这个药的临床应用"时需要检索多次 |
| **检索精度下降** | 用户问"人参的性味归经"时，检索到的 chunk 可能包含含量测定的无关内容 | 噪声多，回答质量差 |

### 三级结构化语义切分策略

本项目采用**三级结构化语义切分**策略，尊重药典数据的天然语义边界：

```
Level 1: 药品边界
  │  每个药品 / 子剂型 / 饮片为独立处理单位
  │  （由 Parser 阶段完成识别）
  │
  ▼
Level 2: 章节边界
  │  按【性状】【鉴别】【检查】等章节标记分割
  │  （由 Parser 阶段完成识别）
  │
  ▼
Level 3: 智能分组与二次切分（Chunker 核心逻辑）
  │
  ├── (a) 短章节合并 ──────────────────────────────────┐
  │      性味与归经 + 功能与主治 + 用法与用量            │
  │      + 注意 + 贮藏 → 合并为一个"临床应用"chunk      │
  │                                                    │
  ├── (b) 长章节按成分切分                              │
  │      【含量测定】中按"黄苓/金银花/连翘"分别描述      │
  │      → 按成分二次切分为独立 chunk                    │
  │                                                    │
  ├── (c) 超长章节滑动窗口                              │
  │      仍超过阈值的章节                               │
  │      → 按自然句滑动窗口切分，保留 100 字符重叠        │
  │                                                    │
  └── (d) 中等章节保持完整                              │
         性状、鉴别等通常长度适中                        │
         → 直接作为一个 chunk                           │
```

#### 详细处理流程

对每个药品条目，Chunker 按以下流程处理：

```
药品条目进入
    │
    ▼
整体内容 ≤ 1500字符 且 章节数 ≤ 8？
    │ 是
    ├──→ 整个条目作为一个 whole_entry chunk（短条目快速路径）
    │
    │ 否
    ▼
生成药品概要 chunk（summary）
  包含：药品名 + 拼音 + 拉丁名 + 分类 + 概述
    │
    ▼
遍历每个章节，按类型分流：
    │
    ├── 临床短章节（性味归经/功能主治/用法用量/注意/贮藏）
    │   且长度 < 300 字符
    │   → 收集到 clinical_parts，稍后合并
    │
    ├── 详述章节但过短（< 50字符）
    │   → 也合并到 clinical_parts
    │
    ├── 详述章节（性状/鉴别/检查/含量测定/浸出物等）
    │   → 调用 _chunk_detailed_section()
    │       │
    │       ├── 内容 ≤ 1500字符 → 直接成 chunk（detailed）
    │       │
    │       ├── 【含量测定】+ 含多成分 → 按成分切分（assay_component）
    │       │
    │       ├── 【检查】+ 超长 → 按检查项切分（check_item）
    │       │
    │       └── 仍超长 → 滑动窗口切分（window）
    │
    └── 其他章节 → 按详述章节处理
    │
    ▼
合并 clinical_parts → "临床应用" chunk（clinical）
```

#### 各切分策略说明

**（a）短章节合并 — 临床应用 chunk**

药典中以下章节通常很短（几十到一两百字符），单独成 chunk 会过于碎片化：

- `【性味与归经】` — 如"甘、微苦，微温。归脾、肺、心经。"
- `【功能与主治】` — 如"大补元气，复脉固脱，补脾益肺..."
- `【用法与用量】` — 如"3〜9g。"
- `【注意】` — 如"不宜与藜芦同用。"
- `【贮藏】` — 如"置阴凉干燥处，防潮。"

这些章节在语义上紧密关联（都是临床应用信息），合并为一个 chunk 既减少碎片，又保证临床信息的完整性。

**（b）按成分切分 — 含量测定 chunk**

复方制剂的 `【含量测定】` 通常按成分分别描述检测方法，例如双黄连口服液：

```
黄苓  照高效液相色谱法（通则0512）测定...（黄芩苷的完整测定方法）
金银花  照高效液相色谱法（通则0512）测定...（绿原酸的完整测定方法）
连翘  照高效液相色谱法（通则0512）测定...（连翘苷的完整测定方法）
```

通过正则匹配 `成分名 + "照" + 检测方法` 模式，将不同成分的测定方法切分为独立 chunk。这样用户问"双黄连口服液中金银花的含量测定"时，可以精确检索到对应内容。

**（c）滑动窗口切分 — 超长章节**

对于仍超过 1500 字符的章节（如复杂的鉴别实验、特征图谱描述），按自然句切分后使用滑动窗口：

- **窗口大小**：每段不超过 1500 字符
- **重叠区域**：相邻 chunk 保留 100 字符重叠
- **切分依据**：按中文句号 `。`、分号 `；`、感叹号 `！`、问号 `？`、换行符切分

重叠区域确保跨 chunk 的语义连续性，避免关键信息被切分到两个 chunk 的边界处而丢失。

**（d）药品概要 chunk — summary**

对于内容较长的药品条目，额外生成一个概要 chunk，包含药品名、拼音、拉丁名、分类和概述文本。这个 chunk 作为药品的"索引卡片"，在用户进行宽泛查询时（如"人参是什么"）提供快速定位。

### Chunk 类型说明

| chunk_type | section | 说明 | 示例场景 |
|------------|---------|------|----------|
| `whole_entry` | 完整条目 | 短药品条目整体作为一个 chunk | 内容 ≤1500 字符的小型药材 |
| `summary` | 药品概要 | 药品基本信息（名称、拼音、分类、概述） | 为长条目生成索引卡片 |
| `clinical` | 临床应用 | 多个短章节合并的临床信息 | 性味归经+功能主治+用法用量 |
| `detailed` | 对应章节名 | 单个章节完整内容 | 【性状】【鉴别】等中等长度章节 |
| `assay_component` | 含量测定-成分N | 按成分切分的含量测定 | 复方制剂中各成分的测定方法 |
| `check_item` | 检查-项N | 按检查项切分的检查内容 | 水分/总灰分/重金属等检查项 |
| `window` | 章节名-段 | 滑动窗口切分的超长内容 | 复杂的鉴别实验描述 |

### Chunk 元数据说明

每个 Chunk 携带丰富的元数据，用于下游的混合检索和精确过滤：

```json
{
  "chunk_id": "chunk_000005",
  "content": "本品为椭圆形、长椭圆形或不规则的斜切片...",
  "drug_name": "丁公藤-饮片",
  "pinyin_name": "",
  "latin_name": "",
  "category": "药材和饮片",
  "section": "性状",
  "chunk_type": "detailed",
  "is_yinpian": true,
  "is_sub_formulation": true,
  "parent_drug": "丁公藤",
  "table_markdown": null,
  "char_count": 131
}
```

| 字段 | 类型 | 说明 | 检索用途 |
|------|------|------|----------|
| `chunk_id` | str | 全局唯一 ID | 引用追踪 |
| `content` | str | chunk 正文内容 | 向量化 + BM25 索引 |
| `drug_name` | str | 药品名称 | 元数据过滤（按药品检索） |
| `pinyin_name` | str | 拼音名 | 模糊匹配（拼音检索） |
| `latin_name` | str | 拉丁学名 | 精确匹配 |
| `category` | str | 分类 | 元数据过滤（按分类检索） |
| `section` | str | 章节名 | 元数据过滤（按章节检索） |
| `chunk_type` | str | chunk 类型 | 检索策略优化 |
| `is_yinpian` | bool | 是否饮片 | 元数据过滤 |
| `is_sub_formulation` | bool | 是否子剂型 | 元数据过滤 |
| `parent_drug` | str | 父级药品名 | 关联检索 |
| `table_markdown` | str\|null | 关联表格（Markdown） | 表格内容检索 |
| `char_count` | int | 内容字符数 | 质量监控 |

### 切片参数配置

切片相关参数集中在 `src/etl/chunker.py` 和 `src/config.py` 中，可根据实际效果调整：

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| `MAX_CHUNK_CHARS` | 1500 | chunker.py | 单个 chunk 最大字符数 |
| `WINDOW_SIZE` | 600 | chunker.py | 滑动窗口大小 |
| `OVERLAP_CHARS` | 100 | chunker.py | 滑动窗口重叠字符数 |
| `SHORT_SECTION_THRESHOLD` | 300 | chunker.py | 短章节阈值（小于此值视为可合并） |
| `MIN_DETAILED_CHARS` | 50 | chunker.py | 详述章节最小字符数（低于则合并到临床 chunk） |

**章节分组配置**（`chunker.py`）：

- `CLINICAL_SECTIONS`：临床应用类章节（性味与归经、功能与主治、用法与用量、注意、贮藏、规格等）
- `OVERVIEW_SECTIONS`：概述类章节（来源、制法、处方、炮制、提取等）
- `DETAILED_SECTIONS`：详述类章节（性状、鉴别、检查、浸出物、含量测定、特征图谱等）

---

## 输出数据格式

ETL 流水线产出两个 JSON 文件：`drugs.json` 是**中间产物**（按药品组织的完整档案），`chunks.json` 是**最终产物**（拆分成块后的检索单元）。后者是后续向量化和检索的直接输入。

| 文件 | 角色 | 数据粒度 | 用途 |
|------|------|----------|------|
| `drugs.json` | 中间产物 | 按药品组织，保留完整层级结构 | 数据备查、调试、重新切片 |
| `chunks.json` | 最终产物 | 已切分为扁平的检索块 | **直接喂给向量数据库做 Embedding 和检索** |

### drugs.json — 结构化药品条目

这是 **Parser + Cleaner** 阶段的输出，将 `.docx` 文档解析成了结构化的药品数据并完成清洗。每个元素代表一个药品条目（药材 / 饮片 / 子剂型），保留药品的**层级结构**——一个药品下有多个 `sections`，每个 section 有自己的章节名和正文。约 2,700+ 条记录。

**真实数据示例**（以"一枝黄花"为例）：

```json
{
  "drug_name": "一枝黄花",          // 药品名称
  "pinyin_name": "Yizhihuanghua",   // 拼音名
  "latin_name": "SOLIDAGINIS HERBA", // 拉丁学名
  "parent_drug": "",                 // 父级药品（子剂型/饮片才有值，如"一枝黄花"）
  "is_sub_formulation": false,       // 是否子剂型
  "is_yinpian": false,               // 是否饮片
  "category_hint": "药材和饮片",      // 分类（药材和饮片/成方制剂/植物油脂和提取物）
  "intro_text": "本品为菊科植物一枝黄花...的干燥全草。", // 来源/概述
  "sections": [                      // 章节列表（核心内容）
    {
      "section_name": "性状",        // 章节名（性状/鉴别/检查/含量测定等）
      "content": "本品长30〜100cm...",// 章节正文
      "table_markdown": null,        // 关联表格（Markdown格式，无表格则为null）
      "raw_paragraphs": ["【性状】本品长30〜100cm..."] // 原始段落（保留备查）
    },
    {
      "section_name": "浸出物",
      "content": "照水溶性浸出物测定法...不得少于17.0%。",
      "table_markdown": null,
      "raw_paragraphs": ["【浸出物】照水溶性浸出物测定法..."]
    },
    {
      "section_name": "含量测定",
      "content": "照高效液相色谱法（通则0512）测定...",
      "table_markdown": null,
      "raw_paragraphs": ["【含量测定】照高效液相色谱法..."]
    }
  ],
  "para_start": 2,                   // 段落起始索引（原始文档中的位置）
  "para_end": 2                      // 段落结束索引
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `drug_name` | str | 药品名称（如"人参"、"双黄连口服液"、"一枝黄花-饮片"） |
| `pinyin_name` | str | 拼音名（如"Renshen"） |
| `latin_name` | str | 拉丁学名（如"GINSENG RADIX ET RHIZOMA"），仅药材有 |
| `parent_drug` | str | 父级药品名，子剂型/饮片才有值（如饮片"一枝黄花-饮片"的 parent_drug 为"一枝黄花"） |
| `is_sub_formulation` | bool | 是否子剂型（同一药品的不同剂型） |
| `is_yinpian` | bool | 是否饮片 |
| `category_hint` | str | 药品分类：药材和饮片 / 成方制剂和单味制剂 / 植物油脂和提取物 |
| `intro_text` | str | 来源或概述文本（药品名后、第一个章节标记前的内容） |
| `sections` | list | 章节列表，每个元素包含 `section_name`、`content`、`table_markdown`、`raw_paragraphs` |
| `para_start` / `para_end` | int | 原始文档中的段落位置索引（调试用） |

### chunks.json — RAG 检索单元

这是 **Chunker** 阶段的输出，将 `drugs.json` 中的药品条目按结构化语义策略切分成了扁平的检索块。每个元素是一个独立的 Chunk，是后续向量化和检索的直接输入。约 11,056 个。

**真实数据示例**（展示不同 chunk_type）：

**类型 1：完整条目（whole_entry）— 短药品整体成一个 chunk**

```json
{
  "chunk_id": "chunk_000001",
  "content": "本品为菊科植物一枝黄花...的干燥全草。\n【性状】本品长30〜100cm...\n【鉴别】...\n【检查】...\n【浸出物】...\n【含量测定】...",
  "drug_name": "一枝黄花",
  "pinyin_name": "Yizhihuanghua",
  "latin_name": "SOLIDAGINIS HERBA",
  "category": "药材和饮片",
  "section": "完整条目",
  "chunk_type": "whole_entry",
  "is_yinpian": false,
  "is_sub_formulation": false,
  "parent_drug": "",
  "table_markdown": null,
  "char_count": 1320
}
```
> 内容 ≤1500 字符的短药品，整体打包成一个 chunk，保留所有章节信息。

**类型 2：药品概要（summary）— 长药品的"索引卡"**

```json
{
  "chunk_id": "chunk_000004",
  "content": "药品名称：丁公藤-饮片（丁公藤的子剂型）\n分类：药材和饮片",
  "drug_name": "丁公藤-饮片",
  "pinyin_name": "",
  "latin_name": "",
  "category": "药材和饮片",
  "section": "药品概要",
  "chunk_type": "summary",
  "is_yinpian": true,
  "is_sub_formulation": true,
  "parent_drug": "丁公藤",
  "table_markdown": null,
  "char_count": 29
}
```
> 长药品额外生成的概要，包含名称、拼音、分类、概述，方便宽泛查询时快速定位。

**类型 3：单章节详述（detailed）— 中等长度章节独立成 chunk**

```json
{
  "chunk_id": "chunk_000005",
  "content": "本品为椭圆形、长椭圆形或不规则的斜切片，直径l~10cm,厚0.2〜0.7cm...",
  "drug_name": "丁公藤-饮片",
  "pinyin_name": "",
  "latin_name": "",
  "category": "药材和饮片",
  "section": "性状",
  "chunk_type": "detailed",
  "is_yinpian": true,
  "is_sub_formulation": true,
  "parent_drug": "丁公藤",
  "table_markdown": null,
  "char_count": 131
}
```
> 【性状】、【鉴别】等中等长度章节，独立成一个 chunk，保证语义完整。

**字段说明**：

| 字段 | 类型 | 说明 | 检索用途 |
|------|------|------|----------|
| `chunk_id` | str | 全局唯一 ID（如 chunk_000005） | 引用追踪 |
| `content` | str | chunk 正文内容 | 向量化 + BM25 索引 |
| `drug_name` | str | 药品名称 | 元数据过滤（按药品检索） |
| `pinyin_name` | str | 拼音名 | 模糊匹配（拼音检索） |
| `latin_name` | str | 拉丁学名 | 精确匹配 |
| `category` | str | 分类 | 元数据过滤（按分类检索） |
| `section` | str | 章节名（性状/鉴别/含量测定/临床应用等） | 元数据过滤（按章节检索） |
| `chunk_type` | str | chunk 类型（见下表） | 检索策略优化 |
| `is_yinpian` | bool | 是否饮片 | 元数据过滤 |
| `is_sub_formulation` | bool | 是否子剂型 | 元数据过滤 |
| `parent_drug` | str | 父级药品名 | 关联检索 |
| `table_markdown` | str\|null | 关联表格（Markdown 格式） | 表格内容检索 |
| `char_count` | int | 内容字符数 | 质量监控 |

**chunk_type 类型说明**：

| chunk_type | 说明 | 示例场景 |
|------------|------|----------|
| `whole_entry` | 短药品条目整体作为一个 chunk | 内容 ≤1500 字符的小型药材 |
| `summary` | 药品基本信息（名称、拼音、分类、概述） | 为长条目生成索引卡片 |
| `clinical` | 多个短章节合并的临床信息 | 性味归经+功能主治+用法用量+注意+贮藏 |
| `detailed` | 单个章节完整内容 | 【性状】【鉴别】等中等长度章节 |
| `assay_component` | 按成分切分的含量测定 | 复方制剂中各成分的测定方法 |
| `check_item` | 按检查项切分的检查内容 | 水分/总灰分/重金属等检查项 |
| `window` | 滑动窗口切分的超长内容 | 复杂的鉴别实验、特征图谱描述 |

### 两个文件的关系

```
drugs.json（按药品组织的完整档案）
    │
    │  Chunker 按结构化语义策略切分
    │
    ▼
chunks.json（拆分成块的检索单元）
    │
    │  阶段二：对每个 chunk 的 content 做 Embedding + BM25 索引
    │
    ▼
向量数据库 + BM25 索引 + SQLite 元数据 → 两路检索 + RRF 融合 → LLM 生成回答
```

- `drugs.json` 是"按药品整理好的完整档案"，保留药品 → 章节 → 段落的层级结构，适合人工查阅、调试和重新切片
- `chunks.json` 是"把档案拆成一块一块的检索单元"，是扁平结构，每个 chunk 都携带药品元数据，是向量化和检索的直接输入

**当前数据规模**（2020年药典一部）：

| 指标 | 数量 |
|------|------|
| 药品条目数（drugs.json） | 约 2,700+ 条 |
| Chunk 总数（chunks.json） | 约 11,056 个 |
| 平均每药品 chunk 数 | 约 4.1 个 |

---

## 索引构建（阶段二）

### 混合检索架构

药典检索场景同时需要**语义理解**和**精确匹配**，单一检索方式无法兼顾：

| 检索方式 | 优势 | 劣势 | 药典场景示例 |
|----------|------|------|-------------|
| **向量检索** | 语义理解强，能匹配同义/近义表达 | 专业术语精确匹配弱 | "人参的性味归经" → 匹配到含"性味与归经"的 chunk |
| **BM25 检索** | 精确匹配术语、编号、数值 | 无法理解语义 | "通则0512" → 精确命中含该编号的 chunk |

本项目采用**两路检索 + RRF 融合 + SQLite 辅助**的混合检索架构：

```
用户查询
    │
    ├──【检索路 1】向量检索（FAISS）     → Top-15 语义匹配结果 ─┐
    │                                                          │
    ├──【检索路 2】BM25 检索（jieba+BM25）→ Top-15 关键词匹配结果 ─┤
    │                                                          ▼
    │                                                  RRF 融合 → Top-N
    │                                                          │
    └──【辅助存储】SQLite ──────────── 按 chunk_id 补全元数据 ◄┘
         （不产生排名，不参与融合）
```

**三个索引各司其职**：

| 索引 | 角色 | 产生排名？ | 说明 |
|------|------|-----------|------|
| **FAISS 向量索引** | 检索路 1 | ✅ 参与 RRF | 语义匹配，理解同义/近义表达 |
| **BM25 关键词索引** | 检索路 2 | ✅ 参与 RRF | 精确匹配术语、编号、数值 |
| **SQLite 元数据库** | 辅助存储 | ❌ 不参与 | 检索后补全元数据 + 结构化精确过滤 |

> SQLite 不参与排名融合，而是承担两个辅助职责：
> 1. **检索后补全元数据** — RRF 融合结果只有 `chunk_id` 和 `content`，通过 SQLite 按 ID 批量查出药品名、章节、分类等完整信息
> 2. **结构化精确过滤** — 支持 SQL 级别的复杂查询（如 `drug_name='人参' AND section='含量测定'`），弥补向量检索过滤能力不足

### 向量索引（FAISS）

**文件**: `src/indexing/embedder.py` + `src/indexing/vector_index.py`

使用 `BAAI/bge-large-zh-v1.5` 模型将 chunk 文本编码为 1024 维归一化向量，存入 FAISS `IndexFlatIP` 索引。

> **为什么用 FAISS 而非 ChromaDB？**
> ChromaDB 在 Windows 上存在 HNSW 索引持久化问题（0.5.x 无法持久化，1.x Rust 后端加载失败），
> FAISS 是 Facebook 出品的成熟向量检索库，Windows 兼容性完美，且 `IndexFlatIP` + 归一化向量 = 精确余弦相似度检索。

**BGE 模型特点**：

- 中文语义理解能力强（C-MTEB 榜单排名靠前）
- 输出 1024 维归一化向量（配合余弦相似度）
- 编码 query 时需添加指令前缀：`"为这个句子生成表示以用于检索相关文章：" + query`
- 编码文档时不加前缀

**Embedder 封装**（`embedder.py`）：

```python
from indexing.embedder import Embedder

embedder = Embedder()                          # 自动检测 GPU/CPU

# 批量编码文档（不加前缀）
doc_vectors = embedder.encode_documents(texts)  # → (N, 1024) 归一化向量

# 编码查询（加 BGE 指令前缀）
query_vec = embedder.encode_query("人参的性味归经")  # → (1024,) 归一化向量
```

**FAISS 索引**（`vector_index.py`）：

```python
from indexing.vector_index import FaissVectorIndex

index = FaissVectorIndex()

# 写入
index.add_documents(ids, embeddings, documents, metadatas)

# 语义检索（支持元数据过滤）
results = index.query(query_embedding, top_k=15, where={"drug_name": "人参"})
# → [{"id": "chunk_000123", "content": "...", "metadata": {...}, "score": 0.92}, ...]
```

### 关键词索引（BM25）

**文件**: `src/indexing/keyword_index.py`

使用 `jieba` 分词（内置药典自定义词典）+ `rank-bm25` 构建全文关键词索引。

**为什么需要自定义词典？**

药典领域有很多专业术语不应被 jieba 默认词典切分：

| 原始文本 | jieba 默认分词 | 自定义词典分词 |
|----------|---------------|---------------|
| 双黄连口服液 | 双 / 黄连 / 口服 / 液 | **双黄连** / **口服液** |
| 人参皂苷Rg1 | 人参 / 皂 / 昔 / Rg1 | **人参皂苷** / Rg1 |
| 高效液相色谱法 | 高效 / 液 / 相 / 色谱 / 法 | **高效液相色谱法** |
| 通则0512 | 通则 / 0512 | **通则0512** |

**自定义词典来源**：

1. **硬编码领域词汇**（`CUSTOM_WORDS`）：检测方法名、通则编号、化学成分名、章节标记等
2. **动态提取药品名**：构建索引时从 `chunks.json` 中提取所有药品名，自动加入 jieba 词典

```python
from indexing.keyword_index import BM25Index, JiebaTokenizer

# 分词测试
tokens = JiebaTokenizer.tokenize("照高效液相色谱法（通则0512）测定人参皂苷Rg1")
# → ['高效液相色谱法', '通则0512', '测定', '人参皂苷', 'Rg1']

# BM25 检索
index = BM25Index()
index.load()  # 加载已构建的索引
results = index.query("人参皂苷含量测定", top_k=15)
# → [{"id": "chunk_000123", "content": "...", "score": 15.2}, ...]
```

### 元数据存储（SQLite）

**文件**: `src/indexing/metadata_store.py`

将 chunk 的**完整元数据**（13 个字段）存入 SQLite，是检索流程中的辅助存储。

> **注意**：SQLite 不参与 RRF 排名融合，它承担两个辅助职责：

**职责 1：检索后补全元数据**

RRF 融合后的结果只有 `chunk_id` 和 `content`，需要通过 SQLite 按 ID 批量查出药品名、章节、分类等完整信息，用于结果展示和下游生成：

```python
# 融合结果只有 id 和 content
fused = rrf_fusion([vector_results, bm25_results], top_n=5)

# 用 SQLite 按 ID 批量补全元数据
top_ids = [r['id'] for r in fused]
meta_map = {m['chunk_id']: m for m in store.get_by_ids(top_ids)}
# → {"chunk_000123": {"drug_name": "人参", "section": "含量测定", ...}, ...}
```

**职责 2：结构化精确过滤**

支持 SQL 级别的复杂查询，弥补向量检索元数据过滤能力不足（FAISS 不支持 LIKE、OR 等）：

```python
from indexing.metadata_store import MetadataStore

store = MetadataStore()

# 按药品名 + 章节精确过滤
results = store.filter(drug_name="人参", section="含量测定")

# 按分类过滤
results = store.filter(category="成方制剂和单味制剂")

# 获取所有药品名（用于前端药品列表）
drug_names = store.get_all_drug_names()  # → ["一枝黄花", "丁公藤", ...]

# 统计信息
stats = store.stats()
# → {"total_chunks": 11056, "total_drugs": 1381, ...}
```

**数据库索引**：为 `drug_name`、`category`、`section`、`chunk_type`、`parent_drug` 字段创建了 B-tree 索引，保证过滤查询毫秒级响应。

### RRF 多路融合

**文件**: `src/indexing/fusion.py`

使用 RRF (Reciprocal Rank Fusion) 算法融合**两路检索**（向量 + BM25）的结果。SQLite 不参与融合。

**RRF 算法原理**：

对每个文档，计算它在所有检索路径中排名的倒数之和：

$$\text{score}(d) = \sum_{i} \frac{1}{k + \text{rank}_i(d)}$$

其中 $k$ 是平滑参数（默认 60），$\text{rank}_i(d)$ 是文档 $d$ 在第 $i$ 路检索中的排名。

**RRF 优势**：

- 不依赖各路检索的原始分数（向量相似度 vs BM25 分数量纲不同）
- 只看排名，天然适合融合不同检索器的结果
- 简单高效，无需训练

```python
from indexing.fusion import rrf_fusion

# 融合两路检索结果
fused = rrf_fusion(
    rank_lists=[vector_results, bm25_results],
    k=60,       # 平滑参数
    top_n=30,   # 返回 Top-30
)
# → [{"id": "chunk_000123", "rrf_score": 0.0308, "sources": ["vector", "bm25"], ...}, ...]
```

### 索引构建脚本

**文件**: `scripts/build/build_index.py`

一键构建索引的脚本，构建三套独立的索引存储：

```
chunks.json（11,056 个 chunk）
    │
    ├──[1/3] SQLite 元数据索引  ──→ metadata.db        （~0.3s）
    │
    ├──[2/3] BM25 关键词索引    ──→ bm25_index.pkl     （~9s）
    │       └─ 从数据中提取 1,359 个药品名 → 加入 jieba 词典
    │
└──[3/3] FAISS 向量索引      ──→ faiss/             （~5min GPU）
    └─ BGE-large-zh 编码 → 1024 维归一化向量 → 写入 FAISS IndexFlatIP
```

**命令行参数**：

```bash
python scripts/build/build_index.py [--skip-embedding] [--test]
```

| 参数 | 说明 |
|------|------|
| `--skip-embedding` | 跳过向量编码（仅构建 BM25 + SQLite，用于快速测试） |
| `--test` | 构建完成后运行混合检索测试（4 个示例查询） |

### 索引配置参数

索引相关参数集中在 `src/config.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-large-zh-v1.5` | HuggingFace 模型名 |
| `EMBEDDING_BATCH_SIZE` | 32 | 批量编码大小 |
| `EMBEDDING_DIM` | 1024 | 向量维度 |
| `BGE_QUERY_INSTRUCTION` | `为这个句子生成表示以用于检索相关文章：` | 查询编码前缀指令 |
| `EMBEDDING_DIM` | 1024 | FAISS 索引维度 |
| `CHROMA_DISTANCE_METRIC` | `cosine` | 距离度量（配合归一化向量+IndexFlatIP） |
| `VECTOR_TOP_K` | 15 | 向量检索召回数 |
| `BM25_TOP_K` | 15 | BM25 检索召回数 |
| `RRF_K` | 60 | RRF 融合平滑参数 |

### 混合检索验证

构建完成后，使用 4 个典型查询验证混合检索效果：

| 查询 | 验证点 | 效果 |
|------|--------|------|
| 人参的性味归经是什么？ | 语义检索 + 关键词检索互补 | 向量匹配到含"性味""归经"的 chunk，BM25 精确匹配"人参" |
| 双黄连口服液的含量测定方法 | 复方制剂按成分切分的精确检索 | 成功召回双黄连系列药品及高效液相色谱法相关内容 |
| 哪些药材含有挥发油成分？ | BM25 关键词精确匹配 | 精准召回沙苑子、石决明、侧柏叶等含挥发油的药材 |
| 高效液相色谱法的操作步骤 | 方法类查询的语义+关键词双匹配 | 召回多个含色谱法操作流程的药品条目 |

---

## 检索引擎（阶段三）

在阶段二构建的三路索引基础上，实现完整的检索引擎，将查询解析、多路召回、RRF 融合、重排模型整合为统一接口。

### 检索流程总览

```
  用户查询："人参的性味归经是什么？"
    │
    ▼
  ┌─────────────────────────────────────────────────────────┐
  │  1. 查询解析（QueryParser）                               │
  │     药品名=["人参"] 章节=["性味与归经"] 意图="属性查询"  │
  │     filter={"drug_name":"人参"}                            │
  └────────────────────┬────────────────────────────────────┘
                       ▼
  ┌──────────────┬──────────────────────────────────────────┐
  │ 2a. 向量检索  │ 2b. BM25 检索                              │
  │ (FAISS)      │ (jieba+BM25Okapi)                          │
  │ top_k=15     │ top_k=15                                   │
  │ +元数据过滤   │                                            │
  └──────┬───────┴──────┬─────────────────────────────────────┘
         │              │
         ▼              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  3. RRF 融合 (k=60) → top_n=30 候选                      │
  └────────────────────┬────────────────────────────────────┘
                       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  4. 重排（BGE-Reranker-v2-m3）→ top_k=5 最终结果        │
  └────────────────────┬────────────────────────────────────┘
                       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  5. 元数据补全（SQLite）+ 上下文组装 → 返回              │
  └─────────────────────────────────────────────────────────┘
```

### 查询解析器（QueryParser）

**文件**: `src/retrieval/query_parser.py`

对用户自然语言查询进行结构化解析，提取三类信息：

| 解析维度 | 说明 | 示例 |
|----------|------|------|
| **药品名识别** | 从 SQLite 加载所有药品名，在查询中做子串匹配 | "人参的性味归经" → `["人参"]` |
| **章节识别** | 基于预定义章节标记 + 同义词映射 | "怎么服用" → `["用法与用量"]` |
| **查询意图** | 基于规则匹配，判断信息类别 | "性味归经" → `属性查询` |

支持 10 种查询意图分类：属性查询、功效查询、用法查询、鉴别查询、检测查询、炮制查询、性状查询、储藏查询、制剂查询、成分查询。

解析结果用于：
- 为向量检索添加元数据过滤条件（`where={"drug_name": "人参"}`）
- 为上下文组装提供结构化信息

### 重排模型（Reranker）

**文件**: `src/retrieval/reranker.py`

使用 **BAAI/bge-reranker-v2-m3** 对 RRF 融合后的候选结果进行精排。

| 对比项 | Embedding（双塔） | Reranker（交叉编码器） |
|--------|-------------------|----------------------|
| query/doc 处理 | 独立编码 | 拼接编码，全注意力交互 |
| 速度 | 快（可离线编码） | 慢（必须在线计算） |
| 精度 | 中 | 高 |

**为什么需要重排？**
- 向量检索速度快但精度有限，召回的 Top-30 中可能有噪声
- BM25 基于词频匹配，无法理解语义
- RRF 融合只看排名，不评估真正的相关性
- 重排模型将 query 和 document 拼接输入，能精确评估语义相关性

### 主检索引擎（Retriever）

**文件**: `src/retrieval/retriever.py`

统一协调所有检索组件，提供简洁的 API 接口：

```python
from retrieval.retriever import Retriever

# 初始化（自动加载所有索引和模型）
retriever = Retriever()

# 完整检索
response = retriever.search("人参的性味归经是什么？")
print(response.context)          # 上下文文本（供 LLM 生成用）
print(response.latency)          # 检索耗时
print(response.parsed)           # 查询解析结果
for r in response.results:       # 检索结果
    print(r.drug_name, r.section, r.rerank_score)

# 简化接口
results = retriever.search_simple("人参的功效")  # → List[Dict]
context = retriever.get_context("人参的功效")     # → str
```

**返回数据结构**：

- `SearchResult`：单条检索结果，包含 chunk_id、content、score、drug_name、section 等完整元数据
- `RetrievalResponse`：完整响应，包含查询解析、结果列表、上下文文本、各阶段耗时

### 上下文组装

检索引擎自动将 Top-K 结果组装为 LLM 可用的上下文文本：

```
【检索结果 1】药品：人参 | 章节：性味与归经
人参 甘、微苦，微温。归脾、肺、心经。
---
【检索结果 2】药品：人参 | 章节：功能与主治
大补元气，复脉固脱，补脾益肺，生津养血，安神益智。
---
```

组装策略：
- 每个 chunk 内容截断到 `CONTEXT_MAX_CHARS_PER_CHUNK`（800字）
- 上下文总长度不超过 `CONTEXT_MAX_TOTAL_CHARS`（4000字）
- 自动添加药品名、章节等结构化标签

### 检索引擎配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RERANKER_MODEL_PATH` | `D:\MODEL\BAAI\bge-reranker-v2-m3` | 重排模型路径 |
| `RERANKER_TOP_K` | 5 | 重排后返回的最终结果数 |
| `RERANKER_MAX_LENGTH` | 512 | 重排模型最大序列长度 |
| `RRF_TOP_N` | 30 | RRF 融合后保留的候选数 |
| `ENABLE_RERANKER` | True | 是否启用重排 |
| `CONTEXT_MAX_CHARS_PER_CHUNK` | 800 | 上下文中单个 chunk 最大字符数 |
| `CONTEXT_MAX_TOTAL_CHARS` | 4000 | 上下文总字符数上限 |

### 检索引擎使用示例

**测试脚本**: `scripts/test/test_retrieval.py`

```bash
python scripts/test/test_retrieval.py
```

测试覆盖 6 种查询类型：

| 查询 | 类型 | 验证点 |
|------|------|--------|
| 人参的性味归经是什么？ | 属性查询 | 精确药品名 + 明确章节 |
| 双黄连口服液的含量测定方法 | 检测查询 | 复方制剂 + 含量测定 |
| 哪些药材含有挥发油成分？ | 成分查询 | 无具体药品名，全库检索 |
| 高效液相色谱法的操作步骤 | 方法查询 | 检测方法类问题 |
| 黄芪和当归一起用的功效 | 功效查询 | 多药品名 |
| 六味地黄丸怎么服用？ | 用法查询 | 成方制剂 + 用法用量 |

---

## 生成引擎（阶段四）

在阶段三检索引擎基础上，集成美团 **LongCat 2.0** 大语言模型，实现端到端的药典智能问答。包含 Prompt 工程、LLM 生成、回答后处理和多轮对话管理。

### 生成流程总览

```
  用户查询："人参的性味归经是什么？"
    │
    ▼
  ┌─────────────────────────────────────────────────────────┐
  │  1. 指代消解（DialogueManager）                           │
  │     多轮对话时：将"它"→"人参"（通过 LLM 推断）           │
  └────────────────────┬────────────────────────────────────┘
                       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  2. 检索（Retriever）                                     │
  │     查询解析 → 向量+BM25 → RRF → 重排 → 上下文组装        │
  └────────────────────┬────────────────────────────────────┘
                       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  3. Prompt 构建（Prompts）                                │
  │     System Prompt + 药典参考资料 + 对话历史 + 用户问题     │
  └────────────────────┬────────────────────────────────────┘
                       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  4. LLM 生成（LLMClient → LongCat 2.0）                  │
  │     基于检索上下文生成专业回答                             │
  └────────────────────┬────────────────────────────────────┘
                       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  5. 后处理（PostProcessor）                               │
  │     引用标注 + 一致性校验 + 格式美化 + 用药安全提醒        │
  └────────────────────┬────────────────────────────────────┘
                       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  6. 更新对话历史（DialogueManager）                       │
  └─────────────────────────────────────────────────────────┘
```

### LLM 客户端（LLMClient）

**文件**: `src/generation/llm_client.py`

封装美团 LongCat 2.0 API（OpenAI 兼容接口）的调用逻辑。

**LongCat 2.0 API 信息**：

| 项目 | 值 |
|------|-----|
| 接入端点 | `https://api.longcat.chat/openai` |
| 模型名称 | `LongCat-2.0` |
| API 格式 | OpenAI 兼容（`/v1/chat/completions`） |
| 上下文长度 | 1M tokens |
| 最大输出 | 128K tokens |

**特性**：

- 使用 OpenAI Python SDK 调用，无需额外 SDK
- 自动重试（指数退避，最多 3 次）
- 支持流式输出（`chat_stream`）
- API Key 从环境变量 `LONGCAT_API_KEY` 读取，避免硬编码

```python
from generation.llm_client import LLMClient

client = LLMClient()  # 自动从环境变量读取 API Key

# 非流式调用
answer = client.chat(messages)

# 流式调用
for chunk in client.chat_stream(messages):
    print(chunk, end="", flush=True)
```

### Prompt 模板设计

**文件**: `src/generation/prompts.py`

为药典问答场景设计专用的 Prompt 模板，核心原则是**严格基于检索到的药典原文回答**，防止 LLM 幻觉。

**System Prompt 规则**：

1. 只能基于【药典参考资料】中的内容回答，不得添加参考资料以外的信息
2. 如果参考资料中没有涉及用户问题的内容，明确回答"药典中未收录相关信息"
3. 引用药典原文时用引号标注，并注明出处
4. 保持专业、准确、简洁，使用药典规范术语
5. 涉及用药建议时提醒"具体用药请遵医嘱"
6. 多个检索结果涉及同一问题时综合归纳
7. 使用 Markdown 格式输出

**Prompt 结构**：

```
<|system|> 药典专家角色定义 + 7 条回答规则
<|user|>   药典参考资料（检索结果注入）+ 用户问题
```

多轮对话时在 System 和 User 之间插入历史对话消息。

### 回答后处理（PostProcessor）

**文件**: `src/generation/postprocessor.py`

对 LLM 生成的回答进行四步后处理：

| 步骤 | 功能 | 说明 |
|------|------|------|
| 1. 格式美化 | 清理多余空白、统一格式 | 合并连续空行、去除行尾空格 |
| 2. 引用标注 | 自动添加来源标注 | `[1] 药典2020一部-人参-性味与归经` |
| 3. 一致性校验 | 检查数值一致性 | 回答中的剂量/含量数值是否在检索原文中出现 |
| 4. 安全提醒 | 用药安全提示 | 涉及用法用量时自动添加"具体用药请遵医嘱" |

```python
from generation.postprocessor import PostProcessor

processor = PostProcessor()
result = processor.process(raw_answer, retrieval_results)
print(result.answer)              # 后处理后的回答
print(result.citations)           # 引用来源列表
print(result.consistency_issues)  # 一致性问题（空列表=无问题）
```

### 多轮对话管理（DialogueManager）

**文件**: `src/generation/dialogue.py`

管理多轮对话上下文，支持指代消解：

```
第1轮  用户: "人参的性味归经是什么？"
       系统: [回答人参性味归经]

第2轮  用户: "那它的功能主治呢？"  ← 指代消解: "它"="人参"
       系统: [回答人参功能主治]

第3轮  用户: "用法用量是怎样的？"  ← 上下文延续
       系统: [回答人参用法用量]
```

**指代消解流程**：

1. 检测用户查询中是否包含指代词（"它"、"这个药"、"该药材"等）
2. 如有，构建指代消解 Prompt，将对话历史和当前查询发送给 LLM
3. LLM 返回消解后的完整查询（"那它的功能主治呢？" → "那人参的功能主治呢？"）
4. 使用消解后的查询进行检索

**上下文窗口控制**：

- 保留最近 5 轮对话（`DIALOGUE_MAX_HISTORY`）
- 对话历史总字符数不超过 4096（`DIALOGUE_CONTEXT_WINDOW`）

### 主生成引擎（Generator）

**文件**: `src/generation/generator.py`

整合检索引擎 + LLM 生成 + 后处理，提供端到端的问答能力。

```python
from generation.generator import Generator

# 初始化（自动加载检索引擎 + LLM 客户端 + 对话管理器）
generator = Generator()

# 单轮问答
response = generator.answer("人参的性味归经是什么？")
print(response.answer)           # 最终回答（经后处理）
print(response.citations)        # 引用来源
print(response.latency)          # 总耗时
print(response.component_latency) # 各阶段耗时

# 多轮对话（自动指代消解）
generator.answer("人参的性味归经是什么？")
response = generator.answer("那它的功能主治呢？")
# "它" 自动消解为 "人参"

# 流式输出
for chunk in generator.answer_stream("当归的功效是什么？"):
    print(chunk, end="", flush=True)

# 清空对话历史
generator.clear_dialogue()
```

**返回数据结构**：

- `GenerationResponse`：完整生成响应，包含原始查询、消解后查询、最终回答、LLM 原始回答、引用来源、一致性问题、各阶段耗时、对话轮次

### 生成引擎配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LONGCAT_API_KEY` | 环境变量 | LongCat API Key |
| `LONGCAT_BASE_URL` | `https://api.longcat.chat/openai` | API 端点 |
| `LONGCAT_MODEL` | `LongCat-2.0` | 模型名称 |
| `LLM_TEMPERATURE` | 0.3 | 低温度保证准确性 |
| `LLM_MAX_TOKENS` | 2048 | 单次回答最大 token 数 |
| `LLM_TOP_P` | 0.9 | 核采样参数 |
| `LLM_MAX_RETRIES` | 3 | API 调用失败重试次数 |
| `DIALOGUE_MAX_HISTORY` | 5 | 多轮对话保留的最大轮数 |
| `DIALOGUE_CONTEXT_WINDOW` | 4096 | 对话历史最大字符数 |
| `ENABLE_CITATION` | True | 是否启用引用标注 |
| `ENABLE_CONSISTENCY_CHECK` | True | 是否启用一致性校验 |
| `ENABLE_FORMAT_BEAUTIFY` | True | 是否启用格式美化 |

### 生成引擎使用示例

**测试脚本**: `scripts/test/test_generation.py`

```bash
# 设置 API Key
$env:LONGCAT_API_KEY="your_api_key_here"

# 运行测试
python scripts/test/test_generation.py
```

测试覆盖三类场景：

| 测试 | 内容 | 验证点 |
|------|------|--------|
| 单轮问答 | 4 种查询类型 | 属性查询、检测查询、功效查询、用法查询 |
| 多轮对话 | 3 轮连续对话 | 指代消解（"它"→"人参"） |
| 流式输出 | 实时生成 | 逐 token 输出 |

---

## API 服务（阶段五）

基于 FastAPI 构建 RESTful API，对外提供药典智能问答服务。支持多轮对话、流式输出、检索查询和药品管理。

### API 端点总览

| 方法 | 路径 | 功能 | 标签 |
|------|------|------|------|
| `POST` | `/api/v1/chat` | 智能问答（支持多轮对话） | 问答 |
| `POST` | `/api/v1/chat/stream` | 流式问答（SSE 实时输出） | 问答 |
| `GET` | `/api/v1/search` | 检索查询（支持药品/章节过滤） | 检索 |
| `GET` | `/api/v1/drugs` | 药品列表查询 | 药品 |
| `POST` | `/api/v1/sessions` | 创建会话 | 会话 |
| `GET` | `/api/v1/sessions` | 列出活跃会话 | 会话 |
| `GET` | `/api/v1/sessions/{id}` | 获取会话详情 | 会话 |
| `DELETE` | `/api/v1/sessions/{id}` | 删除会话 | 会话 |
| `GET` | `/api/v1/health` | 健康检查 | 系统 |
| `GET` | `/api/v1/stats` | 系统统计信息 | 系统 |
| `GET` | `/docs` | Swagger UI 交互式文档 | - |

### 智能问答接口

**`POST /api/v1/chat`**

```json
// 请求体
{
  "question": "人参的性味归经是什么？",
  "session_id": null,           // 可选，多轮对话会话 ID
  "return_sources": true,       // 是否返回检索来源
  "temperature": 0.3,           // 可选，LLM 温度参数
  "max_tokens": 2048            // 可选，最大输出 token 数
}

// 响应体
{
  "answer": "根据药典记载，人参的性味为甘、微苦，微温...",
  "session_id": "sess_abc123",
  "sources": [
    {
      "chunk_id": "chunk_000123",
      "drug_name": "人参",
      "section": "性味与归经",
      "content": "甘、微苦，微温。归脾、肺、心经。",
      "score": 0.95,
      "rerank_score": 0.92
    }
  ],
  "citations": ["药典2020一部-人参-性味与归经"],
  "resolved_query": "人参的性味归经是什么？",
  "latency_ms": 8481,
  "dialogue_turn": 1,
  "consistency_issues": []
}
```

**多轮对话示例**：

```python
import requests

# 第1轮
r = requests.post("http://localhost:8000/api/v1/chat", json={
    "question": "当归的性味和功效是什么？"
})
session_id = r.json()["session_id"]

# 第2轮（复用 session_id，自动指代消解）
r = requests.post("http://localhost:8000/api/v1/chat", json={
    "question": "那它的用法用量呢？",
    "session_id": session_id
})
# "它" 自动消解为 "当归"
```

### 流式问答接口

**`POST /api/v1/chat/stream`**

使用 Server-Sent Events (SSE) 格式逐 token 返回生成内容：

```python
import requests

r = requests.post("http://localhost:8000/api/v1/chat/stream", json={
    "question": "黄芪有什么功效？"
}, stream=True)

for line in r.iter_lines():
    if line:
        data = json.loads(line.decode("utf-8").replace("data: ", ""))
        if "content" in data:
            print(data["content"], end="", flush=True)
        elif data.get("done"):
            print("\n[完成]")
```

### 检索查询接口

**`GET /api/v1/search?q=关键词&drug=药品名&section=章节&top_k=5`**

```bash
# 检索含"黄芪"的内容
curl "http://localhost:8000/api/v1/search?q=黄芪+功效&top_k=3"

# 按药品名过滤
curl "http://localhost:8000/api/v1/search?q=含量测定&drug=人参&top_k=5"
```

### 药品列表接口

**`GET /api/v1/drugs?keyword=关键词&category=分类&limit=100`**

```bash
# 搜索含"人参"的药品
curl "http://localhost:8000/api/v1/drugs?keyword=人参&limit=10"

# 获取所有成方制剂
curl "http://localhost:8000/api/v1/drugs?category=成方制剂和单味制剂"
```

### 会话管理接口

| 操作 | 方法 | 路径 |
|------|------|------|
| 创建会话 | `POST` | `/api/v1/sessions` |
| 列出会话 | `GET` | `/api/v1/sessions` |
| 会话详情 | `GET` | `/api/v1/sessions/{id}` |
| 删除会话 | `DELETE` | `/api/v1/sessions/{id}` |

### 系统接口

```bash
# 健康检查
curl http://localhost:8000/api/v1/health
# → {"status": "ok", "model": "LongCat-2.0", "chunks_count": 11056, ...}

# 统计信息
curl http://localhost:8000/api/v1/stats
# → {"total_chunks": 11056, "total_drugs": 1381, "by_category": {...}, ...}
```

### API 启动与配置

**启动方式**：

```bash
# 设置 API Key
$env:LONGCAT_API_KEY="your_api_key_here"

# 启动 API 服务
conda activate llm
python scripts/run/run_api.py

# 或直接使用 uvicorn
cd src
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**启动参数**：

```bash
python scripts/run/run_api.py [--host 0.0.0.0] [--port 8000] [--reload]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8000` | 监听端口 |
| `--reload` | False | 开发模式（热重载） |

启动后访问：
- API 文档（Swagger UI）：`http://localhost:8000/docs`
- 交互式 API 测试：`http://localhost:8000/redoc`

**测试脚本**：`scripts/test/test_api.py`

```bash
# 启动 API 服务后，另开终端运行
python scripts/test/test_api.py
```

---

## Web UI（阶段六）

基于 Streamlit 构建的可视化交互界面，为药典智能问答系统提供用户友好的操作入口。

### 技术架构

```
src/webui/
  __init__.py       # 模块入口
  app.py             # 主应用（页面路由、侧边栏、5 个功能页面）
  api_client.py      # API HTTP 客户端（封装所有后端调用）
  components.py      # 可复用 UI 组件（来源卡片、引用标签、统计卡片等）
  styles.py          # CSS 主题样式（中医药风格配色体系）
```

**设计规范**（遵循 UI/UX Pro Max 技能推荐）：

| 维度 | 选型 | 说明 |
|------|------|------|
| 风格 | Accessible & Ethical | WCAG AAA 无障碍标准 |
| 主色 | #2D6A4F 深草药绿 | 代表传统中药 |
| 辅色 | #D4A373 暖琥珀金 | 传统质感 |
| 背景 | #FAF8F1 暖米白 | 柔和不刺眼 |
| 字体 | Noto Sans SC | 中文无障碍字体 |
| 对比度 | ≥ 4.5:1 | 满足正文可读性要求 |
| 触摸目标 | ≥ 44px | 满足移动端可用性 |
| 动画 | 150-300ms | 支持 prefers-reduced-motion |

### 页面功能

| 页面 | 功能说明 |
|------|----------|
| **智能问答** | 流式对话（SSE）、多轮上下文、引用展示、检索来源、用药安全提醒 |
| **检索查询** | 独立检索引擎（向量+BM25+RRF+重排），支持药品/章节过滤 |
| **药品浏览** | 药典收录药品列表，支持模糊搜索 |
| **系统统计** | 索引状态、分类统计、Chunk 类型统计 |
| **关于系统** | 架构说明、技术栈、使用指南 |

### UI 启动与配置

**前提条件**：后端 API 服务已启动（`python scripts/run/run_api.py`）。

```bash
# 方式一：使用启动脚本
python scripts/run/run_webui.py

# 方式二：直接使用 Streamlit CLI
streamlit run src/webui/app.py --server.port 8501
```

启动后访问 http://localhost:8501 即可使用。

**环境变量**：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_BASE_URL` | `http://127.0.0.1:8000` | 后端 API 地址 |

---

## 后续阶段

ETL 流水线（阶段一）、索引构建（阶段二）、检索引擎（阶段三）、生成引擎（阶段四）、API 服务（阶段五）、Web UI（阶段六）已完成，后续规划：

| 阶段 | 内容 | 状态 |
|------|------|------|
| **阶段一：数据构建** | 文档解析 → 清洗 → 智能切片 | ✅ 已完成 |
| **阶段二：索引构建** | Embedding + FAISS + BM25 + SQLite + RRF 融合 | ✅ 已完成 |
| **阶段三：检索引擎** | 查询解析 + 混合检索 + RRF融合 + BGE重排 + 上下文组装 | ✅ 已完成 |
| **阶段四：生成引擎** | Prompt 工程 + LongCat 2.0 LLM + 后处理 + 多轮对话 | ✅ 已完成 |
| **阶段五：API 服务** | FastAPI RESTful API + SSE 流式 + 会话管理 | ✅ 已完成 |
| **阶段六：Web UI** | Streamlit 交互界面（5 页面、流式对话、中医药主题） | ✅ 已完成 |
| **阶段七：部署** | Docker 容器化 | ⏳ 待实现 |

---

## License

本项目数据来源于《中国药典》2020年版，仅供学习和研究使用。
