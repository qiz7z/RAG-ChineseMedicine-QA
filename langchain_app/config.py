# -*- coding: utf-8 -*-
"""
LangChain 标准版配置
====================
完全独立于 src/config.py：路径、LLM、检索参数全部在此定义，
敏感信息（API Key）从环境变量 / 项目根目录 .env 读取。
"""
import os
from pathlib import Path

# 项目根目录（langchain_app/ 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """加载 .env（KEY=VALUE，# 注释）；真实环境变量优先"""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except FileNotFoundError:
        pass


_load_dotenv(BASE_DIR / ".env")

# ------------------------------------------------------------
# 路径（数据文件与主项目共享；代码完全独立）
# ------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
CHUNKS_JSON_PATH = DATA_DIR / "processed" / "chunks.json"

# LangChain 版索引（独立目录）
LC_INDEX_DIR = DATA_DIR / "vectorstore" / "langchain_faiss"
LC_BM25_PATH = DATA_DIR / "vectorstore" / "langchain_bm25.pkl"
LC_DRUG_NAMES_PATH = DATA_DIR / "vectorstore" / "langchain_drug_names.json"
REPORT_DIR = DATA_DIR / "eval" / "reports"
TEST_SET_PATH = DATA_DIR / "eval" / "test_queries.json"

# 本地模型
MODELS_DIR = BASE_DIR / "models"
EMBEDDING_MODEL_PATH = str(MODELS_DIR / "bge-large-zh-v1.5")
EMBEDDING_MODEL_PATH = (
    EMBEDDING_MODEL_PATH if os.path.isdir(EMBEDDING_MODEL_PATH) else "BAAI/bge-large-zh-v1.5"
)
# BGE 查询指令（编码 query 时附加，编码文档时不加；对齐官方模型卡片建议）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIM = 1024

# 重排模型（CrossEncoder）
RERANKER_MODEL_PATH = os.environ.get(
    "RERANKER_MODEL_PATH", r"D:\MODEL\BAAI\bge-reranker-v2-m3"
)
RERANKER_TOP_N = 5

# ------------------------------------------------------------
# LLM（OpenAI 兼容接口；变量名沿用 LONGCAT_* 以与主项目共用 .env）
# ------------------------------------------------------------
LLM_API_KEY = os.environ.get("LONGCAT_API_KEY", "")
LLM_BASE_URL = os.environ.get("LONGCAT_BASE_URL", "https://api.longcat.chat/openai")
LLM_MODEL = os.environ.get("LONGCAT_MODEL", "LongCat-2.0")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2048"))

# ------------------------------------------------------------
# 检索参数
# ------------------------------------------------------------
VECTOR_K = 15          # 向量路召回
BM25_K = 15            # BM25 路召回
RRF_WEIGHTS = (0.5, 0.5)   # EnsembleRetriever 两路权重（内部即 RRF, c=60）
FINAL_TOP_N = 5        # 重排后返回数
