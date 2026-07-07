# -*- coding: utf-8 -*-
"""全局配置（模板文件）

使用方法：
  1. 复制此文件为 config.py:  copy src\config.example.py src\config.py
  2. 填入你自己的 API Key 和模型路径
  3. config.py 已在 .gitignore 中，不会被提交
"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")       # 本地模型目录
RAW_DOCX_PATH = os.path.join(DATA_DIR, "raw", "2020年药典一部.docx")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DRUGS_JSON_PATH = os.path.join(PROCESSED_DIR, "drugs.json")
CHUNKS_JSON_PATH = os.path.join(PROCESSED_DIR, "chunks.json")
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstore")
CHROMA_DIR = os.path.join(VECTORSTORE_DIR, "chroma")  # Chroma 独立子目录
BM25_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "bm25_index.pkl")
SQLITE_DB_PATH = os.path.join(VECTORSTORE_DIR, "metadata.db")

# ============================================================
# 索引构建配置（阶段二）
# ============================================================

# Embedding 模型
# 优先使用本地保存的模型（避免每次从 HuggingFace 下载）
# 本地路径：models/bge-large-zh-v1.5/
LOCAL_MODEL_DIR = os.path.join(MODELS_DIR, "bge-large-zh-v1.5")
EMBEDDING_MODEL_NAME = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else "BAAI/bge-large-zh-v1.5"
EMBEDDING_BATCH_SIZE = 32          # 批量编码大小
EMBEDDING_DIM = 1024               # bge-large-zh-v1.5 输出维度
# BGE 模型的查询前缀指令（编码 query 时添加，编码文档时不加）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# Chroma 向量数据库
CHROMA_COLLECTION_NAME = "pharmacopoeia"
CHROMA_DISTANCE_METRIC = "cosine"  # 余弦相似度（配合归一化向量）

# 检索参数
VECTOR_TOP_K = 15                  # 向量检索召回数
BM25_TOP_K = 15                    # BM25 检索召回数
RRF_K = 60                         # RRF 融合平滑参数
RRF_TOP_N = 30                     # RRF 融合后保留的候选数

# ============================================================
# 重排模型配置（阶段三）
# ============================================================
# BGE-Reranker-v2-m3：跨语言重排模型，支持中英双语
# 使用 CrossEncoder 架构，对 query-document 对进行打分
# 模型路径优先从环境变量 RERANKER_MODEL_PATH 读取
# 如果未设置环境变量，则使用下面的默认路径（请修改为你的本地路径）
RERANKER_MODEL_PATH = os.environ.get(
    "RERANKER_MODEL_PATH",
    r"D:\MODEL\BAAI\bge-reranker-v2-m3",  # ← 修改为你本地的模型路径
)
RERANKER_TOP_K = 5                 # 重排后返回的最终结果数
RERANKER_MAX_LENGTH = 512          # 重排模型最大序列长度

# ============================================================
# 检索引擎配置（阶段三）
# ============================================================
# 是否启用重排（关闭则直接返回 RRF 融合结果）
ENABLE_RERANKER = True
# 上下文组装时每个 chunk 的最大字符数（截断过长内容）
CONTEXT_MAX_CHARS_PER_CHUNK = 800
# 上下文组装时的最大总字符数
CONTEXT_MAX_TOTAL_CHARS = 4000

# 药典章节标记（按出现频率排序）
# 这些是药典中用于标记不同信息段的【】标记
SECTION_MARKERS = [
    "来源", "性状", "鉴别", "检查", "浸出物", "含量测定",
    "性味与归经", "性味", "归经", "功能与主治", "主治",
    "用法与用量", "注意", "炮制", "制法", "处方",
    "贮藏", "规格", "制剂", "提取", "特征图谱", "指纹图谱",
    "用途", "酸碱度",
]

# OCR 错别字映射表（从数据分析中发现）
OCR_ERROR_MAP = {
    "【検查】": "【检查】",
    "【含量測定】": "【含量测定】",
    "〔礎藏〕": "〔贮藏〕",
    "〔疋藏〕": "〔贮藏〕",
    "〔规格（l）〕": "〔规格（1）〕",
}

# 短章节分组：这些章节通常很短，合并为一个"临床应用"chunk
SHORT_SECTIONS_TO_GROUP = [
    "性味与归经", "性味", "归经",
    "功能与主治", "主治",
    "用法与用量",
    "注意",
    "贮藏",
]

# 含量测定中按成分二次切分的关键词模式
# 例如双黄连口服液中"黄苓照高效液相色谱法""金银花照高效液相色谱法""连翘照高效液相色谱法"
ASSAY_COMPONENT_PATTERN = r'^([\u4e00-\u9fa5]{1,6})\s*照\s*(?:高效液相色谱法|薄层色谱法|气相色谱法|紫外.*?分光光度法)'

# 切片参数
MAX_CHUNK_CHARS = 1500      # 单个chunk最大字符数
MIN_CHUNK_CHARS = 20        # 单个chunk最小字符数
OVERLAP_CHARS = 100         # 滑动窗口重叠字符数
WINDOW_SIZE = 500           # 滑动窗口大小

# ============================================================
# 生成引擎配置（阶段四）
# ============================================================

# --- LLM 模型配置（美团 LongCat 2.0，OpenAI 兼容接口）---
# API Key 从环境变量读取，避免硬编码泄露
# 请填入你自己的 API Key：https://longcat.chat/platform/api_keys
LONGCAT_API_KEY = os.environ.get("LONGCAT_API_KEY", "your_api_key_here")
LONGCAT_BASE_URL = os.environ.get("LONGCAT_BASE_URL", "https://api.longcat.chat/openai")
LONGCAT_MODEL = os.environ.get("LONGCAT_MODEL", "LongCat-2.0")

# 生成参数
LLM_TEMPERATURE = 0.3               # 低温度保证回答的准确性和稳定性
LLM_MAX_TOKENS = 2048               # 单次回答最大 token 数
LLM_TOP_P = 0.9                    # 核采样参数

# 重试配置
LLM_MAX_RETRIES = 3                 # API 调用失败最大重试次数
LLM_RETRY_DELAY = 2                 # 重试间隔（秒，指数退避基数）

# --- 对话管理配置 ---
DIALOGUE_MAX_HISTORY = 5            # 多轮对话保留的最大历史轮数
DIALOGUE_CONTEXT_WINDOW = 4096      # 对话历史最大字符数

# --- 回答后处理 ---
ENABLE_CITATION = True              # 是否启用引用标注
ENABLE_CONSISTENCY_CHECK = True     # 是否启用一致性校验
ENABLE_FORMAT_BEAUTIFY = True       # 是否启用格式美化

# --- 提示词中的安全提醒 ---
MEDICAL_DISCLAIMER = "具体用药请遵医嘱。"
