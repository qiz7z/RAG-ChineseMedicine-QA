# -*- coding: utf-8 -*-
"""
生成侧「有据性」判分（LLM-as-judge）
====================================
替代原来两个**不可用**的生成侧指标。原指标的问题：

| 原指标 | 问题 |
|--------|------|
| `citation_rate`（100%） | **恒真指标**。引用标签是后处理直接从检索结果的 metadata 拼出来的，只要检索有结果就必然 100%，与回答质量无关。 |
| `consistency_issue_rate`（11~13%） | 只用正则匹配**带单位的数值**（mg/g/ml/片/%），注释里就写着「药品名检查未启用」。文字性编造、张冠李戴、同义改写全检不出来，是幻觉率的**下界**而非幻觉率。 |
| `avg_keyword_coverage`（72~74%） | 只检查期望关键词有没有出现，不查对错、不查有没有编造。**不能当「准确率」报**（项目书目标是"回答准确率 ≥95%"，与本指标不是一回事）。 |

本模块做的是**逐论断判定**：把回答拆成一条条可验证的论断，对每条论断问
「它能否在本次检索到的来源里找到依据」，输出三分类：

    supported     有据：来源里能直接找到（或可无歧义推出）
    unsupported   无据：来源里找不到依据（= 编造 / 幻觉），但也没被来源否定
    contradicted  矛盾：与来源明确冲突（最严重）

由此得到真实指标：

    hallucination_rate   = (unsupported + contradicted) / 论断总数        ← 幻觉率（不再是下界）
    unsupported_rate     = unsupported / 论断总数
    contradiction_rate   = contradicted / 论断总数
    grounded_answer_rate = 无任何 unsupported/contradicted 的答案占比     ← 「回答有据率」
    citation_support_rate= 有据论断数 / 论断总数                          ← 替代恒真的 citation_rate

⚠️ 这是**LLM 主观判定**，与数值正则不同，会有噪声与模型偏差：
  - 温度固定为 0（`temperature=0.0`）以求可复现
  - 判定结果**逐条留痕**（论断原文 + 标签 + 理由），便于人工抽查复核
  - 换模型/换 prompt 会改变数值，**报告里必须记 `judge_model`**，跨版本不可直接比

设计上与 `evaluator.py` 解耦：本模块只依赖一个「能对话的 llm」对象（鸭子类型，需有
`simple_chat(text) -> str`），便于单测时注入假实现，不触发真实 API。
"""
import json
import re
import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# ============================================================
# 论断切分
# ============================================================

# 引用标签：「（来源：药典2020一部-人参-饮片-性味与归经）」「> 来源：xxx」
_CITATION_RE = re.compile(r"[（(【\[]?\s*(?:来源|出处|依据)\s*[:：][^）)】\]]*[）)】\]]?")
# 行首装饰：标题号(#)、引用(>)、列表符(- * + 1.)
_LINE_LEAD_RE = re.compile(r"^\s*(?:#{1,6}\s*|>\s*|[-*+]\s+|\d+\.\s+)", re.M)
# 行内 Markdown 装饰符
# 行内 Markdown 装饰。⚠️ 这里**不能**按单字符删 `~`：在药典领域 `~`/`〜`/`～`
# 是**剂量区间分隔符**（`6~12g` 是六到十二克，不是六十二克）。早先写成 `[*`_~]`
# 会把「常规剂量为每次6~12克」切成「…612克」，判官据此判定"与来源矛盾"——
# 一天之内制造 7 条假幻觉（占当期失败论断 13%，见缺陷 18）。
# 因此只删**成对的删除线标记** `~~`，单个 `~` 一律保留。
_MD_INLINE_RE = re.compile(r"~~|[*`_]")
# 包裹性引号（模型常把原文用引号括起来，会让论断与来源字面不一致）
_WRAP_QUOTES = "\"'“”‘’「」『』《》"
# 纯表格分隔行
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$", re.M)
# 无信息量的套话（不计入论断）
# 出处/文献声明（整条只是"这话出自哪本书"）——不是可核验的药品论断。
# 实测形态：`根据《中国药典》记载`、`据药典记载`、`《中国药典》2020年版一部`…
# 判分器不可能在**药典正文**里找到"出处是药典"的依据，此前一律判 unsupported。
_ATTRIBUTION_RE = re.compile(
    r"^(?:根据|据|依据|按照|参见|见|详见|依)?\s*"
    r"(?:《[^》]{1,24}》|药典|中国药典|参考资料|参考|资料|文献)\s*"
    r"(?:\d{4}\s*年?版?)?\s*(?:一部|二部|三部|四部)?\s*"
    r"(?:的)?\s*(?:记载|规定|所述|记录|写明|明确|说明|内容)?\s*[：:]?$"
)

# 整行加粗/标题（Markdown `**标签**` / `## 标题`）→ **标签**，不是论断。
_HEADING_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s+.+|\*\*(?P<b1>[^*]{1,60})\*\*\s*[：:]?|_{2}(?P<b2>[^_]{1,60})_{2}\s*[：:]?)\s*$"
)


def _is_label_like(inner: str) -> bool:
    """加粗整行是否只是"标签"：短、无句末标点、无数字。

    ⚠️ 反向保护：`**人参不宜与藜芦同用。**`（带句末标点）、`**用量：6〜12g**`（带数字）
    都是**真论断**，必须留下——所以不能简单地"整行加粗就丢"。
    """
    t = (inner or "").strip().strip("：: ")
    if not t or len(t) > 22:
        return False
    if re.search(r"[。！？!?；;]", t):
        return False
    if re.search(r"\d", t):
        return False
    return True


def _split_kept_lines(answer: str) -> tuple:
    """把回答按行分流：→ (参与判分的文本, 丢掉的标题行数, 丢掉的表格行数)

    表格行（含 2 个以上 `|`）：被 `split_claims` 拍平后列结构丢失，
    判官无法核验（`大小 直径5〜8mm 同药材` 读起来不像人话）→ 不参与判分。
    ⚠️ 这是**覆盖面缺口**：答案里的表格内容从此不由判官核验，故条数如实报出。
    """
    kept, n_head, n_tab = [], 0, 0
    for ln in answer.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if s.count("|") >= 2:
            n_tab += 1
            continue
        m = _HEADING_LINE_RE.match(ln)
        if m:
            inner = m.group("b1") or m.group("b2")
            if inner is None or _is_label_like(inner):   # `## 标题` 一定丢；加粗行看形态
                n_head += 1
                continue
        kept.append(ln)
    return "\n".join(kept), n_head, n_tab


_BOILERPLATE = (
    "遵医嘱", "具体用药", "请在医师", "请在药师", "如有不适", "仅供参考",
    "以上内容", "综上所述", "总结", "注意：以上", "温馨提示",
)


# ---- 非论断（框架句 / 引用行）：不该被当作论断送进判官 ----
# 动机：交叉验证实测发现，判官在判「根据药典参考资料，人参的性味归经如下」
# 这类**元话语**与「[4] 药典2020一部-天麻-含量测定」这类**引用标记**时，
# 两个判官给出相反结论（一个宽松判有据、一个因抄不出原文判无据），
# 白白制造不一致并污染分母。它们本就不承载可验证的事实。
# 以「如下/为/是」结尾：说明句子在引出内容后就断了（冒号换行被切分），
# 本身不承载任何可验证事实。实测例：「根据药典记载，半夏的用法用量为」。
# 锚点是**句首必须是引导词**且**句尾是系词**——「本品为薯蓣科植物」这类
# 真论断不以「根据」开头，不受影响。
_FRAMING_RE = re.compile(r"^(根据|依据|按照|基于|综合|参考)[^。]{0,40}(?:如下?|为|是)[。：:]?$")
_CITE_LINE_RE = re.compile(r"^\[[0-9,\s，]+\]\s*\S")
_SOURCE_TITLE_RE = re.compile(r"药典\s*2020\s*一部\s*[-—]")
# 「资料中未收录X」：这是**关于检索结果的陈述**，不是关于药品的事实论断。
# 它是否属实取决于来源集合里有没有 X——第一判官（语义）能判，引文式判官
# 无法用子串核验「不存在」，硬判必不一致。实测例：
#   「根据药典参考资料，本批次资料中未收录麻黄的注意事项相关信息」
_ABSENCE_RE = re.compile(r"(?:未|没有|不|仅|只)[^。]{0,12}(?:收录|包含|含|提及|提供|涉及)")


# 引导句被冒号/换行截断后，剩下的部分以系词结尾（「当归的性味和用法用量如下」
# 「甘草的来源、性味和功能主治如下」「资料中包含的信息有」）——本身无事实内容。
# 加长度上限（≤26 字）兜底：真正的论断不可能以「为/是/有」收尾。
_TAIL_COPULA_RE = re.compile(r"(?:如下|[为是有])[。：:]?$")
# 拒答/引荐句：「如需了解…请参考…完整条目」——是导航语，不是论断。
_REFERRAL_RE = re.compile(r"如需了解|请参考|请查阅|详见|请咨询")


def _is_non_claim(c: str) -> bool:
    """框架句 / 引用行 / 检索陈述 / 引荐句 → True"""
    if _FRAMING_RE.match(c):
        return True
    if len(c) <= 26 and _TAIL_COPULA_RE.search(c):
        return True
    if _REFERRAL_RE.search(c):
        return True
    if _SOURCE_TITLE_RE.search(c):          # 整条只是来源标题的引用
        return True
    if _CITE_LINE_RE.match(c) and len(c) < 32:
        return True                         # 「[4] 药典2020一部-天麻祛风补片-含量测定」
    if _ABSENCE_RE.search(c):
        return True                         # 「资料中未收录…」→ 检索陈述，非药品论断
    if _ATTRIBUTION_RE.match(c):
        return True                         # 「根据《中国药典》记载」→ 出处声明，非药品论断
    return False


def split_claims(answer: str) -> List[str]:
    """把回答拆成一条条**可验证的论断**。

    做三件事：① 去掉引用标签与 Markdown 装饰；② 按句末标点/分号/换行切句；
    ③ 丢掉套话、纯标题、过短片段。

    Args:
        answer: 模型原始回答（含 Markdown 与引用标签）

    Returns:
        论断列表（保序、去重）
    """
    return split_claims_with_stats(answer)[0]


def split_claims_with_stats(answer: str) -> tuple:
    """`split_claims` 的带统计版本 → (论断列表, 丢掉的标题行数, 丢掉的表格行数)

    拆成两个函数是为了**保持 `split_claims` 的旧接口**（大量测试与调用方在用），
    同时让判分器能把"丢了多少非论断"如实报进报告——**分母要透明**。
    """
    if not answer:
        return [], 0, 0
    text, n_head, n_tab = _split_kept_lines(answer)
    text = _CITATION_RE.sub("", text)
    text = _TABLE_SEP_RE.sub(" ", text)
    text = _LINE_LEAD_RE.sub("", text)          # 去掉行首 # > - 1.
    text = _MD_INLINE_RE.sub("", text)          # 去掉行内 * ` _ ~
    text = text.replace("|", " ")               # 表格竖线→空格，保留单元格内容
    # 切句：句末标点、分号、换行都算边界
    parts = re.split(r"[。！？!?；;\n]+", text)
    claims, seen = [], set()
    for p in parts:
        c = re.sub(r"\s+", " ", p).strip()
        c = c.strip(" ，,、:：.　" + _WRAP_QUOTES)   # 去掉包裹引号与多余标点
        if len(c) < 4:                      # 过短片段（标题残渣、编号）
            continue
        if any(b in c for b in _BOILERPLATE):
            continue
        if _is_non_claim(c):                # 框架句 / 引用行——不是可验证论断
            continue
        if c in seen:                       # 去重（同一句被表格与正文各出现一次）
            continue
        seen.add(c)
        claims.append(c)
    return claims, n_head, n_tab


# ============================================================
# 判定
# ============================================================

LABELS = ("supported", "unsupported", "contradicted")

# 每批判定的论断条数。实测原因：一次判 22 条时 JSON 输出会被 max_tokens 截断，
# 导致整题判定失败；且批量太大时模型对后段论断的注意力明显下降。
CLAIMS_PER_BATCH = 6          # 8 → 6：批越大越容易被 max_tokens 截断，
                              # 漏判（unscored）本次高达 43/863 = 5%，会削弱指标可信度
# 判分用的输出预算（config 的 LLM_MAX_TOKENS=2048 对"逐条给理由"偏小）
MAX_JUDGE_TOKENS = 3000

GROUNDING_PROMPT = """你是中医药典问答的**事实核查员**。下面给你【参考资料】（本次检索到的药典原文片段）和【待核查论断】（某个 AI 回答被拆成的句子）。

请**逐条**判断每个论断：

- `supported`：能在参考资料里直接找到依据，或可由参考资料无歧义地推出
- `unsupported`：参考资料里找不到依据（属于凭空编造），但也没有被资料否定
- `contradicted`：与参考资料明确冲突（例如剂量、性味、归经、功效说反了或数值不符）

判定要求：
1. **只依据【参考资料】判断**，不要用你自己的中医药知识补全或替它辩护。
   资料里没写的，就是 `unsupported`，哪怕你认为它是常识。
2. 数值（剂量/限度/含量）逐位比对，不一致即 `contradicted`。
3. 药品名、章节归属张冠李戴（把 A 药的功效说成 B 药的）记 `contradicted`。
4. 纯格式性、无实质内容的论断（如"以下是详细介绍"）判 `supported`。
5. `reason` 用一句话说明依据（引用资料原文片段或指出缺失点），不超过 30 字。

【参考资料】
{sources}

【待核查论断】
{claims}

只输出 JSON，不要任何解释文字，格式：
{{"judgements":[{{"i":1,"label":"supported","reason":"资料中…"}}]}}
"""
# 注意：prompt 用花括号包 JSON 示例，上面已用 {{ }} 转义


@dataclass
class ClaimJudgement:
    index: int
    claim: str
    label: str
    reason: str = ""
    # 引文核验式判官专用：模型抄出的原文片段 + 是否通过机械核验
    quote: str = ""
    quote_verified: bool = False


@dataclass
class GroundingResult:
    """单条回答的有据性判定结果"""
    n_claims: int = 0
    supported: int = 0
    unsupported: int = 0
    contradicted: int = 0
    judgements: List[ClaimJudgement] = field(default_factory=list)
    judge_model: str = ""
    judge_error: str = ""
    # 引文核验式判官专用：抄了引文但核验不通过（伪造引文）的条数
    fabricated_quotes: int = 0
    # 判分口径：被识别为**非论断**而未参与判分的行数（如实报出，避免"悄悄缩小分母"）
    dropped_headings: int = 0        # 标题行 / 整行加粗的标签
    dropped_table_rows: int = 0      # 表格行（拍平后结构丢失，判不了）
    # 判官漏判（批处理输出被 max_tokens 截断、个别条目没返回）的条数。
    # **不计入幻觉**——那不是"发现编造"，是"没判成"（缺陷 16 实测：
    # 25/74 的 unsupported 是这类，其中包含药典逐字原文）。
    unscored: int = 0

    @property
    def grounded(self) -> bool:
        """该回答是否**无任何编造/矛盾**（「回答有据率」的分子）"""
        return self.n_claims > 0 and self.unsupported == 0 and self.contradicted == 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GroundingResult":
        """从 `to_dict()` 的结果重建——**按字段名拷贝**，新增字段自动跟上。

        ⚠️ 评测器早先在这里**手工列白名单字段**（n_claims/supported/.../judge_error），
        于是缺陷 20 新加的 `dropped_headings` / `dropped_table_rows`
        被**静默丢弃、在报告里恒为 0**——数据明明逐题算对了，聚合出来是 0。
        又是一个"指标悄悄说谎"的实例（缺陷 20b）。白名单换成字段名自省后，
        此类漏拷不会再发生。
        """
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    @property
    def scored_claims(self) -> int:
        """实际被判定的论断数（漏判的 unscored 不进分母）"""
        return self.n_claims - self.unscored

    @property
    def hallucination_rate(self) -> float:
        """幻觉率 = (无据 + 矛盾) / **已判定**论断数（0 时记 0，不参与均值）"""
        return ((self.unsupported + self.contradicted) / self.scored_claims
                if self.scored_claims else 0.0)

    @property
    def citation_support_rate(self) -> float:
        """有据论断占比（分母同幻觉率，剔除漏判）——替代恒真的 `citation_rate`"""
        return self.supported / self.scored_claims if self.scored_claims else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update({
            "grounded": self.grounded,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "citation_support_rate": round(self.citation_support_rate, 4),
        })
        return d


def judge_answer(llm, question: str, answer: str,
                 sources: List[Dict[str, Any]]) -> GroundingResult:
    """便捷入口（每次新建一个 GroundingJudge，适合脚本与一次性调用）"""
    return GroundingJudge(llm).judge(question, answer, sources)


def format_sources(sources: List[Dict[str, Any]], max_each: int = 1600,
                   max_n: int = 6) -> str:
    """把来源排成给 LLM 看的清单。

    Args:
        sources: [{"drug_name":…, "section":…, "content":…}, …]
        max_each: 每条来源最多截取多少字（控制 prompt 长度）
        max_n: 最多用几条来源
    """
    if not sources:
        return "（无参考资料——本次检索没有返回任何内容）"
    lines = []
    for i, s in enumerate(sources[:max_n], 1):
        drug = s.get("drug_name") or "?"
        sec = s.get("section") or "?"
        content = re.sub(r"\s+", " ", (s.get("content") or "")).strip()
        if len(content) > max_each:
            content = content[:max_each] + "…（截断）"
        lines.append(f"[资料{i}] {drug} · {sec}\n{content}")
    return "\n\n".join(lines)


def _extract_json(raw: str) -> Optional[dict]:
    """从模型输出里抠出第一个 JSON 对象（容忍 ```json 围栏与前后废话）"""
    if not raw:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    text = m.group(1) if m else None
    if text is None:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        text = raw[start:end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # 常见退化：尾随逗号
        try:
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", text))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def parse_judgement(raw: str, claims: List[str]) -> GroundingResult:
    """解析 LLM 输出为 GroundingResult。

    两种失败要分开对待：

    - **整体不可解析**（模型没按格式输出）→ 记 `judge_error`，**计数保持为 0**。
      这是*判定失败*，不是幻觉证据；把它算成"全部无据"会**抬高**幻觉率、
      让指标失真。汇总时这类题**整体排除**（见 `aggregate`）。
    - **个别条目缺失**（整体 JSON 合法，但漏了第 i 条）→ 该条记 `unscored`，
      **不计入幻觉**。
      ⚠️ 曾按"保守方向"计成 `unsupported`，实测证明是**测量误差**：
      漏判的根源是批处理输出被 max_tokens 截断（越长的论断越容易被截），
      被漏的论断里包含药典逐字原文（「用于气虚乏力，食少便溏…」）——
      把它们算成幻觉，等于让判分基础设施的缺陷污染被测对象的分数
      （缺陷 16 实测：74 条 unsupported 里 25 条是漏判、37 条是来源截断）。
    """
    res = GroundingResult(n_claims=len(claims))
    obj = _extract_json(raw)
    if obj is None:
        res.judge_error = "无法解析 JSON"
        return res

    by_index: Dict[int, dict] = {}
    items = obj.get("judgements") or obj.get("judgments") or []
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                idx = int(it.get("i") or it.get("index") or 0)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(claims):
                by_index[idx] = it

    for i, claim in enumerate(claims, 1):
        it = by_index.get(i)
        label = (it or {}).get("label", "").strip().lower()
        if label not in LABELS:
            label = "unscored"
            reason = (it or {}).get("reason") or "未返回该条判定"
        else:
            reason = (it or {}).get("reason") or ""
        res.judgements.append(ClaimJudgement(i, claim, label, str(reason)[:60]))
        if label == "supported":
            res.supported += 1
        elif label == "contradicted":
            res.contradicted += 1
        elif label == "unscored":
            res.unscored += 1
        else:
            res.unsupported += 1
    return res


class GroundingJudge:
    """用 LLM 对回答做逐论断有据性判定。

    Args:
        llm: 鸭子类型对象，需实现 `simple_chat(text: str) -> str`
        model_name: 记入报告，便于跨版本区分（prompt/模型不同，数值不可比）
    """

    def __init__(self, llm, model_name: str = ""):
        self.llm = llm
        self.model_name = model_name or self._guess_model(llm)

    @staticmethod
    def _guess_model(llm) -> str:
        for attr in ("model", "model_name"):
            v = getattr(llm, attr, None)
            if isinstance(v, str) and v:
                return v
        return type(llm).__name__

    def _ask(self, prompt: str) -> str:
        """问一次模型。

        优先用 `chat(messages, temperature=0.0, max_tokens=…)`：
        - **temperature=0** 求可复现（config 默认 0.3，判定不该有随机性）
        - **显式加大输出预算**：config 的 `LLM_MAX_TOKENS=2048` 对"逐条给理由"的
          JSON 输出偏小，长回答（实测 22 条论断）会被**截断**成非法 JSON。
        对象没有 `chat` 时退回 `simple_chat`（单测里的假 LLM 就是这种）。
        """
        chat = getattr(self.llm, "chat", None)
        if callable(chat):
            try:
                return chat([{"role": "user", "content": prompt}],
                            temperature=0.0, max_tokens=MAX_JUDGE_TOKENS)
            except TypeError:                       # 假对象/旧接口不接受这些参数
                pass
        return self.llm.simple_chat(prompt)

    def _judge_batch(self, claims: List[str], sources_txt: str,
                     offset: int) -> List[ClaimJudgement]:
        """判定一批论断；**整体不可解析时重试一次**（实测首次失败率不低）"""
        prompt = GROUNDING_PROMPT.format(
            sources=sources_txt,
            claims="\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1)),
        )
        for attempt in range(2):
            p = prompt if attempt == 0 else (
                prompt + "\n\n重要：上一次你的输出不是合法 JSON（或被截断）。"
                         "这次**只输出**一个 JSON 对象，不要任何其他文字、不要代码围栏，"
                         "`reason` 控制在 20 字以内。")
            raw = self._ask(p)                          # 异常由调用方收敛
            res = parse_judgement(raw, claims)
            if not res.judge_error:
                for j in res.judgements:
                    j.index += offset
                return res.judgements
        raise ValueError("无法解析 JSON")

    def judge(self, question: str, answer: str,
              sources: List[Dict[str, Any]]) -> GroundingResult:
        """判定一条回答。任何异常都收敛成 judge_error，不抛出（评测要能跑完）。

        论断按 `CLAIMS_PER_BATCH` 分批判定后合并；**只要有一批彻底失败，
        整题就记 judge_error**（部分结果会让幻觉率口径不一致，不如整体排除）。
        """
        claims, n_head, n_tab = split_claims_with_stats(answer)
        if not claims:
            res = GroundingResult(judge_model=self.model_name)
            res.dropped_headings, res.dropped_table_rows = n_head, n_tab
            res.judge_error = "回答为空或无有效论断"
            return res

        sources_txt = format_sources(sources)
        res = GroundingResult(n_claims=len(claims), judge_model=self.model_name)
        res.dropped_headings, res.dropped_table_rows = n_head, n_tab
        try:
            for start in range(0, len(claims), CLAIMS_PER_BATCH):
                batch = claims[start:start + CLAIMS_PER_BATCH]
                res.judgements.extend(self._judge_batch(batch, sources_txt, start))
        except ValueError as e:
            # 解析失败（两批都试过）——保留干净的原因文案，便于报告里读
            res = GroundingResult(n_claims=len(claims), judge_model=self.model_name)
            res.dropped_headings, res.dropped_table_rows = n_head, n_tab
            res.judge_error = str(e)[:200]
            return res
        except Exception as e:                      # noqa: BLE001 —— 评测不因单题失败中断
            res = GroundingResult(n_claims=len(claims), judge_model=self.model_name)
            res.dropped_headings, res.dropped_table_rows = n_head, n_tab
            res.judge_error = f"{type(e).__name__}: {e}"[:200]
            return res

        for j in res.judgements:
            if j.label == "supported":
                res.supported += 1
            elif j.label == "contradicted":
                res.contradicted += 1
            elif j.label == "unscored":
                res.unscored += 1
            else:
                res.unsupported += 1
            # 引文核验式判官：给了引文但**核验未通过** = 伪造引文。
            # 在这里从逐条判定**派生**（而不是依赖子类自己累加），
            # 因为本方法会重建 GroundingResult，子类累加的计数会丢。
            # 第一判官从不设置 quote，故此值恒为 0 —— 两份报告结构一致。
            if j.quote and not j.quote_verified and j.quote.strip().upper() != "NONE":
                res.fabricated_quotes += 1
        return res



# ============================================================
# 第二判官：引文核验式（独立机制，用于交叉验证）
# ============================================================
# 为什么需要它：`GroundingJudge` 让模型**直接贴标签**，存在自偏好风险——
# 模型可能对"自己那种风格的说法"更宽松。这里换一个**机制不同**的判法：
#
#   不再问"这句话有据吗"，而是要求**逐字抄出**能支撑它的原文片段；
#   然后由**代码**核验该片段是否真的出现在来源里。
#
# 判定完全建立在可机械验证的证据上：
#   - 抄不出片段           → unsupported
#   - 抄出的片段核验不通过 → **伪造引文**（fabricated），按 unsupported 计并单独计数
#   - 片段核验通过         → supported
#
# 这样两件事同时得到：① 一个与第一判官错误相关性低的独立判法；
# ② 一个第一判官产不出的硬信号——`fabricated_quote_rate`。
# ⚠️ 它**不替代**第一判官：抽句式判法在"需要跨句推理才算有据"时会偏严（抄不出直接判无据）。

QUOTE_PROMPT = """你是中医药典问答的**引文核验员**。下面给你【参考资料】（检索到的药典原文片段）和【待核验论断】。

对每条论断：**从【参考资料】里逐字抄出**能支撑它的原文片段（15~60 字）。

规则：
1. **必须逐字原样抄**，不要改写、不要概括、不要补字、不要拼接不相邻的两段。
2. 抄不出就直接写 `NONE`——**宁可写 NONE，也不要自己编一句像原文的话**。
3. 支撑关系要直接：资料里写了「补肾阳，益精血」，才能支撑「肉苁蓉能补肾阳」这样说法的前半。
   资料里没有的内容（哪怕你认为是常识）一律 `NONE`。
4. 数值必须逐位一致才算支撑。

【参考资料】
{sources}

【待核验论断】
{claims}

只输出 JSON，不要任何解释文字：
{{"judgements":[{{"i":1,"quote":"逐字原文片段 或 NONE"}}]}}
"""

# 引文核验时的容错归一化：只处理**表示形式**差异，不动语义。
# 目的是不让「全角数字」「波浪号写法」「空白」这类差异造成假阴性，
# 但**不**把不同的数字或不同的字归一成同一个。
_QUOTE_SPACE_RE = re.compile(r"\s+")
_QUOTE_TILDE_RE = re.compile(r"[〜～~﹏]")
_QUOTE_SEP_RE = re.compile(r"(?:\.\.\.|…)+")
_FULLWIDTH = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def _norm_for_quote(s: str) -> str:
    """引文核验用的归一化：去空白、统一波浪号、全角→半角、统一标点、去装饰符"""
    if not s:
        return ""
    t = _QUOTE_SPACE_RE.sub("", s)
    t = _QUOTE_TILDE_RE.sub("-", t)
    t = t.translate(_FULLWIDTH)
    # 标点**不敏感**：只保留汉字/字母/数字，以及数字相关的 `-` `%`。
    # 理由：实测模型常漏抄或多抄一个逗号，若按"统一标点"处理会造成假阴性；
    # 而保留的字符集合保证了「文字必须逐字连续」这条实质约束不被削弱。
    t = re.sub(r"[^\u4e00-\u9fff0-9A-Za-z\-%]", "", t)
    return t


def verify_quote(quote: str, sources_text: str, min_len: int = 4) -> bool:
    """核验引文是否**真的**出现在来源文本里（归一化后子串匹配）。

    模型有时会把两段不相邻的原文拼起来（用 `…` 分隔），故按分隔符切开后
    **任一段命中**即算通过（每段至少要 `min_len` 个字符，避免"抄两个字蒙对"）。

    ⚠️ `min_len` 的取值是交叉验证**实测校准**的：
    - 8 → 误杀「归脾、肺、心、肾经」(6字)、「置干燥处，防蛀」(6字)
    - 5 → 误杀剂量类「2〜5g」(4字)、「外用适量」(4字)
    - 4 → 仍能挡住一两个字的偶然命中（「微温」「防蛀」）。
    原则：引文逐字出现在来源里就是真证据，下限只防"极短片段蒙对"。
    """
    if not quote or not sources_text:
        return False
    q = quote.strip()
    if not q or q.upper() == "NONE":
        return False
    src = _norm_for_quote(sources_text)
    for seg in _QUOTE_SEP_RE.split(q):
        ns = _norm_for_quote(seg)
        if len(ns) >= min_len and ns in src:
            return True
    return False


def parse_quote_judgement(raw: str, claims: List[str],
                          sources_text: str) -> GroundingResult:
    """解析引文式判官输出并**机械核验**引文。

    与 `parse_judgement` 的两点不同：
    - 标签由**引文是否核验通过**决定，不由模型自称决定；
    - 模型给了引文但核验不通过 → 记 `fabricated_quotes`（伪造引文）。
    """
    res = GroundingResult(n_claims=len(claims))
    obj = _extract_json(raw)
    if obj is None:
        res.judge_error = "无法解析 JSON"
        return res

    by_index: Dict[int, dict] = {}
    items = obj.get("judgements") or obj.get("judgments") or []
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                idx = int(it.get("i") or it.get("index") or 0)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(claims):
                by_index[idx] = it

    for i, claim in enumerate(claims, 1):
        it = by_index.get(i)
        quote = str((it or {}).get("quote") or "").strip()
        if it is None:
            # 缺条目 = 判官漏判（输出截断），与「明确抄不出」是两回事
            res.unscored += 1
            res.judgements.append(
                ClaimJudgement(i, claim, "unscored", "未返回该条判定", "", False))
            continue
        if not quote or quote.upper() == "NONE":
            res.unsupported += 1
            res.judgements.append(
                ClaimJudgement(i, claim, "unsupported", "未提供引文", quote, False))
            continue
        if verify_quote(quote, sources_text):
            res.supported += 1
            res.judgements.append(
                ClaimJudgement(i, claim, "supported", "引文核验通过", quote, True))
        else:
            # 关键：模型"抄"了但抄不出来 → 伪造引文，按无据计
            res.unsupported += 1
            res.fabricated_quotes += 1
            res.judgements.append(
                ClaimJudgement(i, claim, "unsupported", "引文核验失败（伪造）", quote, False))
    return res


class QuoteVerifiedJudge(GroundingJudge):
    """第二判官：引文核验式。接口与 `GroundingJudge` 完全一致，可直接替换。"""

    def _judge_batch(self, claims: List[str], sources_txt: str,
                     offset: int) -> List[ClaimJudgement]:
        prompt = QUOTE_PROMPT.format(
            sources=sources_txt,
            claims="\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1)),
        )
        for attempt in range(2):
            p = prompt if attempt == 0 else (
                prompt + "\n\n重要：上一次输出不是合法 JSON（或被截断）。"
                         "这次**只输出** JSON，`quote` 控制在 40 字以内。")
            raw = self._ask(p)
            res = parse_quote_judgement(raw, claims, sources_txt)
            if not res.judge_error:
                for j in res.judgements:
                    j.index += offset
                return res.judgements
        raise ValueError("无法解析 JSON")

def aggregate(results: List[GroundingResult]) -> Dict[str, Any]:
    """把逐题判定汇总成报告里的指标（论断级加权，不是答案级平均）。

    **`judge_error` 的题整体排除**：那是"判定失败"，不是"发现幻觉"。
    计入会抬高幻觉率、让指标失真。排除的题数在 `excluded_judge_errors` 里如实报出，
    便于判断本次结果的可信度（排除比例高就该重跑或换模型）。
    """
    ok = [r for r in results if not r.judge_error]
    total = sum(r.n_claims for r in ok)
    # ⚠️ 分母必须与逐题口径一致：剔掉 `unscored`（判官漏判的论断）。
    # 早先聚合处误用 `total`（含 unscored），与 `GroundingResult.hallucination_rate`
    # 的 `scored_claims` 分母不一致——对外报的 6.19% 其实按 840 算，
    # 按文档规定口径（818）应为 6.36%（见缺陷 18b：**指标会为坏分母说谎**）。
    scored = total - sum(r.unscored for r in ok)
    sup = sum(r.supported for r in ok)
    uns = sum(r.unsupported for r in ok)
    con = sum(r.contradicted for r in ok)
    unsc = sum(r.unscored for r in ok)
    judged = [r for r in ok if r.n_claims > 0]
    return {
        "claims_total": total,
        "supported": sup,
        "unsupported": uns,
        "contradicted": con,
        # 幻觉率按**论断级**加权：长篇回答理应贡献更大，
        # 避免"多写一句就多一分风险"被答案级平均掩盖
        "hallucination_rate": round((uns + con) / scored, 4) if scored else 0.0,
        "unsupported_rate": round(uns / scored, 4) if scored else 0.0,
        "contradiction_rate": round(con / scored, 4) if scored else 0.0,
        "citation_support_rate": round(sup / scored, 4) if scored else 0.0,
        # 答案级：完全没有编造/矛盾的答案占比
        "grounded_answer_rate": round(
            sum(1 for r in judged if r.grounded) / len(judged), 4) if judged else 0.0,
        "judged_answers": len(judged),
        "excluded_judge_errors": len(results) - len(ok),
        # 引文核验式判官专用（第一判官恒为 0）
        "unscored_claims": unsc,
        # 判分口径：因"不是论断"而未参与判分的行数——报出来，让分母可审计
        "dropped_headings": sum(r.dropped_headings for r in ok),
        "dropped_table_rows": sum(r.dropped_table_rows for r in ok),
        "scored_claims": scored,   # 幻觉率等比率的分母（如实报出）
        "fabricated_quotes": sum(r.fabricated_quotes for r in ok),
    }
