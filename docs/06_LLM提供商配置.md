# 06 · LLM 提供商配置与切换

> 本项目所有 LLM 调用（手撕版 `LLMClient` 与 LangChain 版 `ChatOpenAI`）都走 **OpenAI 兼容接口 + 环境变量**，切换提供商只需改 `.env` 里的三行，零代码改动。

## 当前配置（2026-08-30）

| 配置项 | 环境变量 | 当前值 |
|---|---|---|
| API 地址 | `LONGCAT_BASE_URL` | `https://apihub.agnes-ai.com/v1` |
| 模型名 | `LONGCAT_MODEL` | `agnes-2.5-flash` |
| API Key | `LONGCAT_API_KEY` | **存放在项目根目录 `.env` 文件中（已被 .gitignore 排除，不进仓库）** |

> ⚠️ 安全约定：**任何密钥都不要写进 `.py` 或 `.md` 等会被 git 跟踪的文件**。密钥只放 `.env`（已忽略）或系统环境变量。

## 切换提供商的三步操作

编辑项目根目录 `.env`：

```ini
LONGCAT_API_KEY=<你的 Key>
LONGCAT_BASE_URL=<OpenAI 兼容的 base_url>
LONGCAT_MODEL=<模型名>
```

然后重启服务即可。两套引擎同时生效：

```bash
python scripts/run/run_lc_api.py     # LangChain 版 (8001)
python scripts/run/run_api.py        # 手撕版 (8000)
```

评估脚本同理（`run_eval.py` / `run_eval_lc.py` 均读取同一组变量）。

## 提供商记录

| 提供商 | Base URL | 模型 | 状态 |
|---|---|---|---|
| Agnes AI APIHub（免费） | `https://apihub.agnes-ai.com/v1` | `agnes-2.5-flash` | **当前使用** |
| 美团 LongCat 开放平台 | `https://api.longcat.chat/openai` | `LongCat-2.0` | 备用（2026-08-30 额度不足 402，Key 建议吊销更换） |

获取新 Key：LongCat → https://longcat.chat/platform/api_keys

## 验证切换是否生效

```bash
# 1. 直接探测（打印模型名即成功）
python -c "import sys; sys.path.insert(0,'src'); from generation.llm_client import LLMClient; print(LLMClient().chat([{'role':'user','content':'你好，请回复OK'}], max_tokens=10))"

# 2. 起服务看健康检查
curl http://127.0.0.1:8001/api/v1/health   # "model" 字段应显示当前模型名
```

## 相关经验

- LongCat 返回 **HTTP 402（额度不足）**属于不可重试错误，两套引擎均已配置 4xx 快速失败，不会白等重试（见 `llm_client.py` 与 `langchain_app/llm.py`）。
- 免费通道可能有速率限制，跑 100 题生成评估前建议先用 `--limit 5` 试跑。
