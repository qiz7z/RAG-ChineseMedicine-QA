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

# 重排模型（CrossEncoder）：默认指向项目内 models/，不再硬编码 Windows 盘符
RERANKER_MODEL_PATH = os.environ.get(
    "RERANKER_MODEL_PATH", str(MODELS_DIR / "bge-reranker-v2-m3")
)
RERANKER_TOP_N = 5
RERANKER_MAX_LENGTH = int(os.environ.get("RERANKER_MAX_LENGTH", "512"))

# 是否启用重排（与主项目同名同义，默认关闭）
# 理由同主项目：本数据集上 CrossEncoder 重排为负收益（见 docs/05、docs/07），
# 且部署环境未下载重排模型，开启会因模型缺失而启动失败。
# 需要复现"开重排"对照时：设 ENABLE_RERANKER=1，或显式传
# build_hybrid_retriever(enable_reranker=True) / 评测脚本去掉 --no-rerank。
ENABLE_RERANKER = os.environ.get("ENABLE_RERANKER", "0").strip().lower() in ("1", "true", "yes")

# 是否启用「章节感知召回」（与主项目同名同义）
# 把正文含【目标章节】标记的候选提前，判据与评测 strict 口径一致。
# 这是标准版此前与手撕版 strict 指标差距的主因之一（标准版原先没有章节解析）。
ENABLE_SECTION_BOOST = os.environ.get(
    "ENABLE_SECTION_BOOST", "1"
).strip().lower() in ("1", "true", "yes")

# ------------------------------------------------------------
# 计算设备
# ------------------------------------------------------------
# 留空 / auto = 自动（有 CUDA 就用 CUDA），行为与改造前一致。
# 低显存机型（如 8GB 笔记本卡）上 embedding(1.3GB) + reranker(2.27GB) 同时驻留
# CUDA 会把显存顶到上限、触发驱动级崩溃；此时可设 RAG_DEVICE=cpu 规避。
# 设备只影响延迟，不影响排序结果，因此评测口径不受影响。
_rag_device_env = os.environ.get("RAG_DEVICE", "").strip().lower()
RAG_DEVICE = None if _rag_device_env in ("", "auto") else _rag_device_env

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
