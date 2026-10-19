# 06 · LLM 提供商配置与切换

> 本项目所有 LLM 调用（手撕版 `LLMClient` 与 LangChain 版 `ChatOpenAI`）都走 **OpenAI 兼容接口 + 环境变量**，切换提供商只需改 `.env` 里的三行，零代码改动。

## 当前配置

| 配置项 | 环境变量 | 当前值 |
|---|---|---|
| API 地址 | `LLM_BASE_URL` | `https://apihub.agnes-ai.com/v1` |
| 模型名 | `LLM_MODEL` | `agnes-2.5-flash` |
| API Key | `LLM_API_KEY` | **存放在项目根目录 `.env` 文件中（已被 .gitignore 排除，不进仓库）** |

> ⚠️ 安全约定：**任何密钥都不要写进 `.py` 或 `.md` 等会被 git 跟踪的文件**。密钥只放 `.env`（已忽略）或系统环境变量。

> 兼容说明：旧变量名 `LONGCAT_*` 仍可继续使用（代码会回退读取），新配置建议统一用 `LLM_*`。

## 切换提供商的三步操作

编辑项目根目录 `.env`：

```ini
LLM_API_KEY=<你的 Key>
LLM_BASE_URL=<OpenAI 兼容的 base_url>
LLM_MODEL=<模型名>
```

然后重启服务即可。两套引擎同时生效：

```bash
python scripts/run/run_lc_api.py     # LangChain 版 (8001)
python scripts/run/run_api.py        # 手撕版 (8000)
```

评估脚本同理（`run_eval.py` / `run_eval_lc.py` 均读取同一组变量）。

## 可选提供商

只要提供 OpenAI 兼容接口（`/chat/completions`），都可以直接接入，例如：

| 类型 | Base URL 示例 | 模型示例 |
|---|---|---|
| 云端 API | `https://<provider>/v1` | 由服务商提供 |
| 本地推理（vLLM / Ollama / LM Studio） | `http://127.0.0.1:8000/v1` | 本地加载的模型名 |

## 验证切换是否生效

```bash
# 1. 直接探测（打印模型名即成功）
python -c "import sys; sys.path.insert(0,'src'); from generation.llm_client import LLMClient; print(LLMClient().chat([{'role':'user','content':'你好，请回复OK'}], max_tokens=10))"

# 2. 起服务看健康检查
curl http://127.0.0.1:8000/api/v1/health   # "model" 字段应显示当前模型名
```

## 相关经验

- 配额/鉴权类错误（HTTP 401/402/403）属于不可重试错误，两套引擎均已配置 4xx 快速失败，不会白等重试（见 `llm_client.py` 与 `langchain_app/llm.py`）。
- 免费通道可能有速率限制，跑 100 题生成评估前建议先用 `--limit 5` 试跑。
