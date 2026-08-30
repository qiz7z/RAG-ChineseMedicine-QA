# 07 · LangChain 标准版架构（完全独立实现）

> 2026-08-30 · 定位：可上线的标准 LangChain 工程，与手撕版（`src/`，保留作对照实验）**零代码依赖**。
> 前一版"自定义适配版"（import src/ 模块的过渡实现）已归档至 `langchain_app/_archive_custom/`，其对比数据见 [docs/05](05_LangChain对比实验.md)。

## 设计原则

1. **只用官方组件**：检索融合、会话记忆、守卫分支、提示词模板全部使用 LangChain 1.x 官方抽象，自定义代码只保留生产实践中正常的两块胶水（查询理解、元数据过滤接入）。
2. **完全独立**：`langchain_app/` 不 import `src/` 的任何模块；`config.py`/`schemas.py`/评估器均为独立实现。共享的只有数据资产（chunks.json、.env）与产品行为（API Schema、提示词规则、守卫词表同源）。
3. **可独立交付**：整个 `langchain_app/` 目录 + `requirements-langchain.txt` + `data/` 即可单独部署。

## 技术栈对照

| 环节 | 标准组件 | 备注 |
|---|---|---|
| 向量库 | `langchain_community.vectorstores.FAISS` | IndexFlatIP + 归一化，独立索引目录 |
| 关键词检索 | `langchain_community.retrievers.BM25Retriever` | jieba 分词（`preprocess_func`），pickle 持久化 |
| 多路融合 | `langchain_classic.retrievers.EnsembleRetriever` | **内置 RRF**，两路 weights 0.5/0.5 |
| 重排 | `HuggingFaceCrossEncoder`（bge-reranker-v2-m3）+ 自实现 `BgeRerankerCompressor` | 该版本组合缺 `CrossEncoderReranker` 类，按标准 `BaseDocumentCompressor` 接口实现 30 行 |
| 生成 | `ChatOpenAI`（OpenAI 兼容端点） | LCEL：`prompt \| llm \| StrOutputParser()` |
| 守卫 | `RunnableBranch` 短路 + 关键词快通道 + LLM 语义兜底 | 词表与主项目同源 |
| 多轮 | `RunnableWithMessageHistory` + condense-question 改写 | per-session `InMemoryChatMessageHistory`，天然隔离 |
| 提示词 | `ChatPromptTemplate` + `MessagesPlaceholder` | 规则与主项目同源 |
| 元数据过滤 | `vectorstore.as_retriever(search_kwargs={"filter": fn})` + BM25 结果后过滤 | `PharmacopoeiaRetriever`（自定义 BaseRetriever，查询级动态过滤） |

## 主链结构（chains.py）

```python
RunnableWithMessageHistory(                      # 按 session_id 注入对话历史
    RunnableLambda(guard_step)                   # 关键词 + LLM 语义守卫
    | RunnableBranch(                            # 守卫短路
        (lambda s: s["guard_rejected"], RunnablePassthrough()),
        RunnableLambda(_core),                   # RAG 核心 ↓
      ),
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# _core 内部：
#   condense-question 改写(多轮) → retriever.invoke → RAG_PROMPT | llm | parser
#   → postprocess（引用标注 + 安全提醒）→ {answer, citations, sources, ...}
```

## 目录结构

```
langchain_app/
├── config.py              # 独立配置（路径/LLM/检索参数，读 .env）
├── llm.py                 # ChatOpenAI 封装
├── embeddings.py          # BGEQueryEmbeddings（查询端附加 BGE 官方指令）
├── build_index.py         # FAISS + BM25Retriever + drug_names.json 构建
├── query_understanding.py # 规则式查询理解（药品名/变体扩展/横向分类）
├── retrievers.py          # PharmacopoeiaRetriever + 标准混合管线 + 重排压缩器
├── guard.py               # 领域守卫（关键词 + LLM 兜底）
├── prompts.py             # ChatPromptTemplate 集
├── postprocess.py         # 引用标注/资料格式化/来源转换
├── chains.py              # ★ 主程序：声明式会话式 RAG 链
├── service.py             # ChatService（answer/answer_stream/search/会话/统计）
├── schemas.py             # API Pydantic 模型（独立副本，与主项目 Schema 一致）
├── api.py                 # FastAPI（端口 8001，前端零改动切换）
├── eval.py                # 独立评估器（判分公式与主项目一致）
└── _archive_custom/       # 旧自定义适配版存档（不参与运行）
```

## 工程踩坑记录

1. **`sys.modules` 污染**：启动脚本若先 import 了 `src/config.py`（同名为 `config`），后续 langchain_app 的 `import config` 全部命中缓存——`run_lc_api.py` 已移除 src 路径，只加 `langchain_app/`。
2. **pydantic 私有属性**：LangChain 组件都是 pydantic 模型，字段名以下划线开头会变成 `PrivateAttr` 占位符（永远 truthy），不能当普通属性用——CrossEncoder 单例已改为模块级变量。
3. **EnsembleRetriever 位置**：1.x 时代从 `langchain.retrievers` 移至 `langchain_classic.retrievers`（包名 langchain-classic）；BM25Retriever 的官方位置是 `langchain_community.retrievers`。
4. **推理模型兼容**：agnes-2.5-flash 会把 token 预算花在 `reasoning_content`，守卫/改写等小预算调用需给足 max_tokens（已设 512/1024）。
5. **GPU 独占**：评估脚本与 API 服务不要同时运行（两 CUDA 进程并发会让单次检索从 ~1s 恶化到 60s+）。

## 检索评估（100 题，与手撕版同判分公式）

| 引擎 | Hit@1 | Hit@3 | Hit@5 | MRR | P50 延迟 |
|---|---|---|---|---|---|
| 手撕版（当前代码） | 89% | 89% | 89% | 0.8900 | 0.727s |
| LangChain 标准版（完整混合+重排） | 91% | 91% | 92% | 0.9120 | 0.683s |
| **标准版去重排（Ensemble 融合）** | **90%** | **92%** | **94%** | **0.9166** | **0.051s** |
| 标准版去 BM25（纯向量+重排） | 89% | 91% | 92% | 0.9008 | 0.360s |

报告：`eval_lc-std_*.json`、`eval_lc-std-noRerank_*.json`、`eval_lc-std-noBM25_*.json`

### 关键发现：CrossEncoder 重排在本数据集上为负收益

去掉重排器后 Hit@5 **92% → 94%**、MRR 0.9120 → 0.9166，延迟反而从 0.683s 降到 0.051s（13 倍）。原因分析：

1. EnsembleRetriever 的 RRF 融合已经把 BM25 的字面精确信号与向量的语义信号结合得足够好，前 5 名排序质量很高；
2. bge-reranker-v2-m3 对"药典条目段落"这类文本的交叉编码打分与相关性并非单调一致，偶尔把正确条目挤出前 5；
3. 消融数据同时显示元数据过滤的价值巨大：纯向量+过滤+重排（92%）比早前的无过滤纯向量 baseline（86%，docs/05）高 6pp。

**工程结论**：检索组件不是堆得越多越好——上线配置采用 `--no-rerank`（默认保留重排开关，便于在更大测试集上复查该结论）。"用测量代替假设"本身就是这次迁移最有价值的产出之一。

> 生成质量（100 题全量，agnes-2.5-flash，报告 `eval_lc-std_generation_20260830_132115.json`）：
> 引用率 **100%**，关键词覆盖率 60.5%，端到端 P50 = 5.44s / P95 = 23.4s。
> 分类型：横向条件 96.7%、跨品种 85.5%、多属性 70.3%、精确数值 60.0%、方法通则 56.7%、
> 多轮 54.2%、单属性 41.3%、否定性 26.7%。零覆盖 24 题中多为关键词字面误伤（"6〜12g" 全角波浪号）
> 与诚实拒答（药典原文确实缺失该章节），另有已知优化点：BM25 分词未注入药品名领域词典，
> 部分药品的临床条目召回弱于手撕版（见下"已知优化点"）。
>
> **生成侧实验（100 题全量 ×3 组，同通道 agnes-2.5-flash，2026-08-30）**：
>
> | 引擎 | 关键词覆盖率 | 引用率 | 一致性问题率 | P50 延迟 |
> |---|---|---|---|---|
> | 手撕版（同条件对照） | 58.6% | 92% | **12%**（超 10% 目标） | 4.61s |
> | 标准版完整（含重排） | 60.5% | 100% | 未测¹ | 5.44s |
> | 标准版去重排 | 59.2% | 99% | 未测¹ | 5.92s |
>
> ¹ 标准版 postprocess 未实现数值一致性校验（手撕版有），0 问题 ≠ 无问题，口径不同不可直接对比。
> **结论**：① 生成侧重排同样无收益（覆盖率差异在噪声内），"去重排"在检索与生成两侧均成立，最优上线配置不变；
> ② 生成侧标准版覆盖率略高（+2pp）、引用率高 8pp，延迟持平——两版生成质量基本相当；
> ③ 手撕版检出 12% 数值一致性问题，值得将数值校验移植到标准版 postprocess（避免"未测"盲区）。
> 方法注记：手撕版评估的对话历史跨题共享（既有实现），标准版每题独立会话，多轮类题目的条件略有差异。

## 运行方式

```bash
pip install -r requirements-langchain.txt
python langchain_app/build_index.py          # 构建索引（FAISS 已有则复用，BM25 约 11s）
python scripts/run/run_lc_api.py             # API 服务（8001）
python scripts/run/run_eval_lc.py            # 检索评估（--no-rerank / --no-bm25 消融）
# 前端切换：API_BASE_URL=http://127.0.0.1:8001 streamlit run src/webui/app.py
```
