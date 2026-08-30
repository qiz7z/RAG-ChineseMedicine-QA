# 05 · LangChain 对比实验：手撕管线 vs LangChain 重建 vs 纯标准组件

> 实验日期：2026-08-30 · 测试集：100 题（data/eval/test_queries.json）· 评估器：src/eval/evaluator.py（三方共用，口径一致）
>
> ⚠️ 本篇的"LangChain hybrid"为**旧自定义适配版**（复用 src/ 模块的过渡实现，代码已归档至 `langchain_app/_archive_custom/`）。
> 现行标准版（完全独立实现）的架构与最新消融数据见 **[07_LangChain标准版架构](07_LangChain标准版架构.md)**，其最优配置 Hit@5 = 94%。
> 本篇保留价值：三方对照的方法论与"重排负收益"结论的首次发现过程。

## 结论速览

1. **手撕混合检索管线可以无损迁移到 LangChain**：LangChain hybrid 版 Hit@5 = 92%，与手撕版同管线水平持平（MRR 0.9058 vs 0.8900），验证检索管线与框架解耦良好。
2. **混合管线相对纯标准组件有显著增益**：LangChain 开箱即用（纯向量检索）只有 Hit@5 = 86%、MRR = 0.7953。混合管线（BM25 + 元数据过滤 + RRF + 重排）带来 **+6pp Hit@5 / +0.11 MRR**，其中"精确数值查询"一项从 100% 掉到 73.3% 再被混合管线救回，是关键词检索路价值的直接证据。
3. **实验过程中发现并修复了手撕版的一个真实缺陷**：QueryParser 返回的分类简称（"药材"）与 SQLite 实际分类值（"药材和饮片"）不匹配，导致横向条件查询的 8/10 题**两路召回被清空**——手撕版当前代码横向查询实测仅 10%。LangChain 版修复该映射后回升至 40%。

## 三方引擎架构对照

| 维度 | 手撕版 (src/) | LangChain hybrid (langchain_app/) | LangChain standard |
|---|---|---|---|
| 向量库 | 自封装 FAISS IndexFlatIP | `langchain_community` FAISS（独立索引，同 bge-large-zh-v1.5 编码规则） | 同左 |
| 关键词检索 | BM25Okapi (jieba) | 复用手撕版 BM25Index | 无 |
| 查询解析/过滤 | QueryParser + SQLite 元数据过滤 | 复用手撕版组件 | 无 |
| 融合 | rrf_fusion (k=60) | 复用 | 无 |
| 重排 | BGE-Reranker-v2-m3 | 复用 | 无 |
| LLM 调用 | 自封装 OpenAI 兼容客户端 | `langchain_openai.ChatOpenAI`（LongCat 2.0） | 同左 |
| 生成管线 | 手写 Generator 流程 | LCEL（`to_lc_messages \| model \| StrOutputParser()`） | 同左 |
| 会话状态 | 全局单例共享（有并发串写风险） | **按 session_id 隔离**（每会话独立 DialogueManager + GuardChecker） | 同左 |
| Prompt / 后处理 / 守卫 | — | 与手撕版共用同一实现（保证生成侧行为一致） | 同左 |

**公平性保证**：三方共用同一份 chunks 数据、BM25 索引、SQLite 元数据、重排模型、Prompt 模板、后处理器与评估器；Embedding 编码规则完全一致（文档不加指令、查询加 BGE 指令前缀、归一化内积）。唯一变量是检索管线本身。

## 总体检索指标

| 指标 | 手撕版<br>（当前代码实测） | LangChain hybrid | LangChain standard | 手撕版<br>（README/7-06 报告口径） |
|---|---|---|---|---|
| Hit@1 | 89.0% | **90.0%** | 75.0% | 91.0% |
| Hit@3 | 89.0% | **91.0%** | 83.0% | 91.0% |
| Hit@5 | 89.0% | **92.0%** | 86.0% | 91.0% |
| MRR | 0.8900 | **0.9058** | 0.7953 | 0.9100 |
| P50 延迟 | 0.727s | 0.733s | **0.015s** | 0.744s |
| P95 延迟 | 1.354s | 1.359s | **0.017s** | 1.353s |

> 关于 README 口径：README 中的 91% 来自 7 月 6 日的报告（eval_retrieval_20260706_193011.json），而横向条件过滤功能是 7 月 7 日加入的——该功能因分类映射缺陷实际降低了横向查询召回，当前代码全量复测为 89%。

## 分类型 Hit@5

| 查询类型 | n | 手撕版（当前） | LC hybrid | LC standard |
|---|---|---|---|---|
| 单药品单属性查询 | 25 | 96% | 96% | 92% |
| 单药品多属性查询 | 15 | 93.3% | 93.3% | 100% |
| 精确数值查询 | 15 | 100% | 100% | **73.3%** |
| 跨品种比较查询 | 10 | 100% | 100% | 100% |
| 横向条件查询 | 10 | **10%** | **40%** | 30% |
| 方法通则查询 | 10 | 100% | 100% | 100% |
| 多轮对话 | 10 | 100% | 100% | 100% |
| 否定性问题 | 5 | 100% | 100% | 80% |

### 分析

- **精确数值查询（-26.7pp）**：如"人参皂苷 Rg1 总量不得少于多少"，查询词与原文数值段落字面重叠度高，BM25 关键词路可直接命中；纯向量检索对"数值+通则编号"类文本的语义表征偏弱。这是混合检索价值最直接的证据。
- **横向条件查询（10% → 40%）**：缺陷定位过程——
  1. 手撕版 `_detect_category_filter` 返回简称 `"药材"`/`"成方制剂"`；
  2. SQLite 实际分类值为 `"药材和饮片"`/`"成方制剂和单味制剂"`；
  3. 测试集 10 道横向题中 8 道被解析出 `category_filter="药材"`，向量路（`{"category": "药材"}` where 过滤）与 BM25 路（SQLite 后过滤）**双双匹配 0 条**，召回被清空；
  4. LangChain 版增加 `CATEGORY_FILTER_MAP` 映射后两路过滤正常工作，回升至 40%（n=10，样本小，且评估器对无 expected_drugs 的题有"检索到即命中"的宽松口径，绝对值偏高，三方同口径相对可比）。
- **单药品多属性（LC-standard 100% > hybrid 93.3%）**：n=15 小样本波动，不具统计显著性。
- **延迟**：hybrid 的 P50（0.733s）约为 standard（0.015s）的 50 倍，代价主要在重排模型前向（~0.6s GPU）与 BM25/融合；换取 +6pp Hit@5 与分类型稳定性。纯向量检索延迟优势在无 GPU 重排场景才会体现。

## 生成质量对比

**10 题抽样**（Agnes 免费通道 agnes-2.5-flash，2026-08-30，报告 `eval_lc_hybrid_generation_20260830_110040.json`）：

| 指标 | 结果 | 说明 |
|---|---|---|
| 引用率 | 100% | 每条回答均标注来源 |
| 一致性问题率 | 0% | 数值无幻觉 |
| 端到端 P50 延迟 | 12.9s | 其中 LLM 13.4s（推理模型思考耗时），检索仅 0.87s |
| 关键词覆盖率 | 48.3% | **指标误伤为主**，见下 |

关键词覆盖率偏低的三个原因（逐题核验后确认回答质量本身正常）：
1. **字符形态差异**：如"当归用法用量"回答"6〜12g"完全正确，但与期望关键词的连字符/波浪号写法不匹配（Q003、Q008）；
2. **已知数据缺口**：川芎原药材在药典 docx 中本就缺失，模型如实回答"未收录"（Q006，README 后续优化方向 #3）；
3. **诚实拒答**：麻黄在药典中无【注意】项，模型正确报告"未提及"而非编造（Q009）。

> 结论：该指标按字面匹配设计，对改写/标点敏感；100 题完整生成对比（含与手撕版同通道对比）待确认正式通道后补充。免费推理通道（agnes-2.5-flash）的守卫语义判定也较慢（思考耗时，单次可达 ~45s；关键词快路径 0ms 不受影响）。

## 工程差异（LangChain 版顺带修复的问题）

| 问题（手撕版） | LangChain 版处理 |
|---|---|
| 全局 Generator 单例，对话历史与守卫信任态被所有用户共享，并发请求互相串写 | `LCGenerator` 按 session_id 持有独立 DialogueManager + GuardChecker，状态天然隔离 |
| `async def` 端点内执行同步重调用（LLM 重试/模型加载），阻塞事件循环等效单并发 | 端点改为同步 `def`，FastAPI 自动入线程池 |
| `/api/v1/search` 的 section 过滤在取回 top_k 之后进行，实际返回数可能远少于请求值 | 超量召回（top_k×4，上限 30）后过滤再截断 |
| LLM 对 402 额度不足等不可重试 4xx 仍重试 3 次（最坏阻塞 14s+） | 4xx（除 429）首次即失败；ChatOpenAI `max_retries=1` |

> 运维注意：Embedding 与重排模型默认加载到 GPU。**评估脚本与 API 服务不要同时运行**——两个 CUDA 进程并发会导致单次检索延迟从 ~1s 恶化到 60s+（本实验实测踩坑）。

## 复现命令

```bash
# 1. 构建 LangChain 版向量索引（GPU 约 4 分钟）
python langchain_app/build_index_lc.py

# 2. 三方检索评估（不需要 LLM 额度）
python scripts/run/run_eval.py --mode retrieval          # 手撕版
python scripts/run/run_eval_lc.py --variant hybrid --mode retrieval
python scripts/run/run_eval_lc.py --variant standard --mode retrieval

# 3. 生成评估（需要 LongCat 额度）
python scripts/run/run_eval.py --mode generation
python scripts/run/run_eval_lc.py --variant hybrid --mode generation

# 4. 启动 LangChain 版服务（8001，与手撕版 8000 并行）
python scripts/run/run_lc_api.py
# 前端切换：API_BASE_URL=http://127.0.0.1:8001 streamlit run src/webui/app.py
```

报告文件：`data/eval/reports/eval_retrieval_20260830_103446.json`（手撕版）、`eval_lc_hybrid_retrieval_20260830_103024.json`、`eval_lc_standard_retrieval_20260830_103039.json`。
