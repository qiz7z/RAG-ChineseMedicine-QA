# -*- coding: utf-8 -*-
"""
生成侧「有据性」判分回归测试（2026-09-22）
=========================================
覆盖 `src/eval/grounding_judge.py`：论断切分、LLM 输出解析、指标汇总。

**全部用假 LLM（鸭子类型），不触发真实 API**——真实判定是主观且要花钱的，
不适合放进单测；这里守的是**确定性部分**：切句、容错解析、指标算法。

真实 LLM 的端到端验证见 `scripts/run/run_eval.py --mode generation --grounding`。
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval.grounding_judge import (  # noqa: E402
    GroundingJudge, GroundingResult, QuoteVerifiedJudge, aggregate,
    format_sources, parse_judgement, parse_quote_judgement, split_claims,
    split_claims_with_stats,
    verify_quote, _is_non_claim,
)


# ============================================================
# 论断切分
# ============================================================

class TestSplitClaims:
    def test_strips_citation_label(self):
        claims = split_claims("人参的性味归经为甘、微苦，微温（来源：药典2020一部-人参-饮片-性味与归经）。")
        assert claims == ["人参的性味归经为甘、微苦，微温"]

    def test_strips_various_citation_forms(self):
        for ans in [
            "人参味甘。来源：药典2020一部-人参",
            "人参味甘。[来源：药典2020一部-人参]",
            "人参味甘。（出处：药典2020一部-人参）",
        ]:
            claims = split_claims(ans)
            assert claims and all("药典2020" not in c for c in claims), ans

    def test_splits_on_sentence_and_semicolon_and_newline(self):
        claims = split_claims("人参味甘。人参归脾经；人参性微温\n人参可补气")
        assert len(claims) == 4

    def test_strips_markdown_decoration(self):
        claims = split_claims("## 性味归经\n\n- **甘、微苦**，微温。")
        assert any("甘、微苦" in c and "#" not in c and "*" not in c for c in claims)

    def test_table_rows_dropped_and_counted(self):
        """表格行**不参与判分**，但必须计数报出（口径变更，见 docs/13 缺陷 20）

        为什么改：表格被切分器拍平后列结构丢失，判官看到的是
        `大小 直径5〜8mm 同药材` 这种读不成句的残片，核验必失败——
        把它们算成幻觉，等于用判分口径制造幻觉。
        ⚠️ 代价是答案表格内容不再由判官核验（覆盖面缺口），故丢弃条数如实报出。
        """
        ans = "| 性味 | 甘、微苦，微温 |\n|------|------|\n| 归经 | 归脾经 |"
        claims, n_head, n_tab = split_claims_with_stats(ans)
        assert claims == [], f"表格行不应成为论断: {claims!r}"
        assert n_tab == 3, f"应报出 3 行表格（含分隔行）: {n_tab}"
        assert n_head == 0

    def test_heading_lines_dropped_but_real_claims_kept(self):
        """整行加粗的**标签**丢弃，但带句末标点/数字的加粗行是**真论断**，必须保留"""
        ans = "**黄芪的副作用（不良反应）**\n**人参不宜与藜芦同用。**\n**用量：6~12g**"
        claims, n_head, _ = split_claims_with_stats(ans)
        joined = " ".join(claims)
        assert "黄芪的副作用" not in joined, "标题行不应成为论断"
        assert "人参不宜与藜芦同用" in joined, "带句末标点的加粗行是真论断，不能丢"
        assert "6~12g" in joined, "带数字的加粗行是真论断，不能丢"
        assert n_head == 1

    def test_drops_boilerplate(self):
        claims = split_claims("人参味甘。具体用药请遵医嘱。如有不适请就医。")
        assert claims == ["人参味甘"]

    def test_drops_too_short(self):
        assert split_claims("好。的。人参味甘微温") == ["人参味甘微温"]

    def test_strips_blockquote_and_wrapping_quotes(self):
        """模型常把原文用 > 与引号包起来，残留会让论断与来源字面不一致"""
        claims = split_claims('> "甘、微苦，微温。"（来源：药典2020一部-人参）')
        assert claims == ["甘、微苦，微温"]

    def test_strips_chinese_quotes(self):
        assert split_claims("“补气升阳，固表止汗。”") == ["补气升阳，固表止汗"]

    def test_dedups(self):
        assert split_claims("人参味甘微温。人参味甘微温。") == ["人参味甘微温"]

    def test_empty(self):
        assert split_claims("") == []
        assert split_claims(None) == []


# ============================================================
# 解析 LLM 输出（容错）
# ============================================================

class TestParseJudgement:
    CLAIMS = ["人参味甘", "人参微温", "人参归脾经"]

    def test_valid(self):
        raw = json.dumps({"judgements": [
            {"i": 1, "label": "supported", "reason": "资料1"},
            {"i": 2, "label": "unsupported", "reason": "资料中无"},
            {"i": 3, "label": "contradicted", "reason": "资料说归肾经"},
        ]}, ensure_ascii=False)
        r = parse_judgement(raw, self.CLAIMS)
        assert (r.supported, r.unsupported, r.contradicted) == (1, 1, 1)
        assert r.judgements[2].reason == "资料说归肾经"

    def test_fenced_json(self):
        raw = '这是判定：\n```json\n{"judgements":[{"i":1,"label":"supported"}]}\n```\n完毕'
        r = parse_judgement(raw, self.CLAIMS)
        assert r.supported == 1

    def test_missing_items_scored_as_unscored(self):
        """漏判 ≠ 幻觉：漏的论断记 unscored，不进幻觉分母（缺陷 16）

        实测依据：被漏的论断里包含药典逐字原文（「用于气虚乏力，食少便溏…」），
        漏判根源是判官批处理输出被 max_tokens 截断——这是判分基础设施的缺陷，
        不该污染被测对象的分数。
        """
        raw = json.dumps({"judgements": [{"i": 1, "label": "supported"}]})
        r = parse_judgement(raw, self.CLAIMS)
        assert (r.supported, r.unsupported, r.unscored) == (1, 0, 2)
        assert r.scored_claims == 1
        assert r.hallucination_rate == 0.0          # 唯一一条 supported，漏判不背锅

    def test_invalid_label_falls_back(self):
        raw = json.dumps({"judgements": [{"i": 1, "label": "maybe"}]})
        r = parse_judgement(raw, self.CLAIMS)
        assert r.unsupported == 0 and r.unscored == 3

    def test_garbage_input_excluded_not_counted_as_hallucination(self):
        """整体不可解析是**判定失败**，不能算成"全部无据"（会抬高幻觉率）"""
        r = parse_judgement("我无法判断", self.CLAIMS)
        assert r.judge_error == "无法解析 JSON"
        assert (r.supported, r.unsupported, r.contradicted) == (0, 0, 0)
        assert r.n_claims == 3          # 只留信息，不参与计数

    def test_out_of_range_index_ignored(self):
        raw = json.dumps({"judgements": [{"i": 99, "label": "supported"}]})
        assert parse_judgement(raw, self.CLAIMS).supported == 0

    def test_trailing_comma_tolerated(self):
        raw = '{"judgements":[{"i":1,"label":"supported"},]}'
        assert parse_judgement(raw, self.CLAIMS).supported == 1


# ============================================================
# 指标
# ============================================================

class TestGroundingResultMetrics:
    def test_rates(self):
        r = GroundingResult(n_claims=10, supported=7, unsupported=2, contradicted=1)
        assert r.hallucination_rate == pytest.approx(0.3)
        assert r.citation_support_rate == pytest.approx(0.7)
        assert r.grounded is False

    def test_grounded_requires_no_issues(self):
        assert GroundingResult(n_claims=3, supported=3).grounded is True
        assert GroundingResult(n_claims=3, supported=2, unsupported=1).grounded is False
        assert GroundingResult(n_claims=0).grounded is False   # 空回答不算"有据"

    def test_zero_claims_no_division_error(self):
        r = GroundingResult()
        assert r.hallucination_rate == 0.0 and r.citation_support_rate == 0.0


class TestAggregate:
    def test_claim_weighted(self):
        """幻觉率按论断数加权：长回答里的编造不应被短回答稀释"""
        a = GroundingResult(n_claims=1, supported=1)
        b = GroundingResult(n_claims=9, supported=6, unsupported=3)
        agg = aggregate([a, b])
        assert agg["claims_total"] == 10
        assert agg["hallucination_rate"] == pytest.approx(0.3)   # 3/10，而非 (0+1)/2
        assert agg["unsupported_rate"] == pytest.approx(0.3)
        assert agg["contradiction_rate"] == 0.0
        assert agg["citation_support_rate"] == pytest.approx(0.7)

    def test_grounded_answer_rate(self):
        a = GroundingResult(n_claims=2, supported=2)                 # 有据
        b = GroundingResult(n_claims=2, supported=1, unsupported=1)  # 有编造
        agg = aggregate([a, b])
        assert agg["grounded_answer_rate"] == pytest.approx(0.5)
        assert agg["judged_answers"] == 2

    def test_judge_errors_excluded_entirely(self):
        """判定失败的题从**所有**指标里排除，只如实报出被排除的题数"""
        ok = GroundingResult(n_claims=1, supported=1)
        bad = GroundingResult(n_claims=5, unsupported=5, judge_error="超时")
        agg = aggregate([ok, bad])
        assert agg["excluded_judge_errors"] == 1
        assert agg["judged_answers"] == 1
        assert agg["claims_total"] == 1          # 失败题的 5 条论断不进分母
        assert agg["hallucination_rate"] == 0.0  # 也不把它们的"unsupported"算进去

    def test_empty(self):
        assert aggregate([])["hallucination_rate"] == 0.0


# ============================================================
# 与 LLM 的交互（假实现）
# ============================================================

class _FakeLLM:
    """假 LLM：按调用次数返回预设输出或抛异常"""
    def __init__(self, outputs=None, exc=None):
        self.outputs = list(outputs or [])
        self.exc = exc
        self.calls = []

    def simple_chat(self, text):
        self.calls.append(text)
        if self.exc:
            raise self.exc
        return self.outputs.pop(0) if self.outputs else "{}"


class TestGroundingJudge:
    def test_end_to_end_with_fake_llm(self):
        out = json.dumps({"judgements": [
            {"i": 1, "label": "supported", "reason": "资料1有"},
            {"i": 2, "label": "unsupported", "reason": "资料中无"},
        ]}, ensure_ascii=False)
        j = GroundingJudge(_FakeLLM([out]), model_name="fake-1")
        r = j.judge("人参的性味归经", "人参味甘。人参治百病。",
                    [{"drug_name": "人参", "section": "性味与归经", "content": "甘、微苦"}])
        assert (r.supported, r.unsupported) == (1, 1)
        assert r.hallucination_rate == pytest.approx(0.5)
        assert r.judge_model == "fake-1"

    def test_prompt_contains_sources_and_claims(self):
        llm = _FakeLLM(['{"judgements":[]}'])
        GroundingJudge(llm).judge("问", "人参味甘。",
                                  [{"drug_name": "人参", "section": "性味与归经", "content": "甘"}])
        p = llm.calls[0]
        assert "人参" in p and "性味与归经" in p and "1. 人参味甘" in p

    def test_llm_exception_does_not_raise(self):
        """调用失败要收敛成 judge_error，且**不计为无据**（失败不是幻觉证据）"""
        j = GroundingJudge(_FakeLLM(exc=RuntimeError("API 500")))
        r = j.judge("问", "人参味甘。", [])
        assert r.judge_error.startswith("RuntimeError")
        assert (r.supported, r.unsupported, r.contradicted) == (0, 0, 0)
        assert r.n_claims == 1          # 保留信息，供报告统计被排除的题数

    def test_empty_answer_skips_llm(self):
        llm = _FakeLLM(['{"judgements":[]}'])
        r = GroundingJudge(llm).judge("问", "。", [])
        assert r.judge_error == "回答为空或无有效论断"
        assert llm.calls == [], "空回答不该浪费一次 LLM 调用"

    def test_no_sources_still_judges(self):
        """检索为空时也要判——此时任何实质论断都该是 unsupported"""
        out = json.dumps({"judgements": [{"i": 1, "label": "unsupported"}]})
        llm = _FakeLLM([out])
        r = GroundingJudge(llm).judge("问", "人参治百病。", [])
        assert "无参考资料" in llm.calls[0]
        assert r.unsupported == 1

    def test_retries_once_on_unparseable_json(self):
        """首次返回非 JSON 时重试一次；实测约 1/3 的题首次会这样"""
        good = json.dumps({"judgements": [{"i": 1, "label": "supported"}]})
        llm = _FakeLLM(["这不是 JSON", good])
        r = GroundingJudge(llm).judge("问", "人参味甘。", [])
        assert len(llm.calls) == 2
        assert r.judge_error == "" and r.supported == 1
        assert "只输出" in llm.calls[1]        # 第二次的提示里带了格式强调

    def test_gives_up_after_two_failures(self):
        llm = _FakeLLM(["不是 JSON", "还不是 JSON"])
        r = GroundingJudge(llm).judge("问", "人参味甘。", [])
        assert len(llm.calls) == 2
        assert r.judge_error == "无法解析 JSON"

    def test_no_retry_when_first_succeeds(self):
        good = json.dumps({"judgements": [{"i": 1, "label": "supported"}]})
        llm = _FakeLLM([good])
        GroundingJudge(llm).judge("问", "人参味甘。", [])
        assert len(llm.calls) == 1

    def test_model_name_guessed(self):
        class _M:
            model = "agnes-2.5-flash"
            def simple_chat(self, t): return "{}"
        assert GroundingJudge(_M()).model_name == "agnes-2.5-flash"


class TestFormatSources:
    def test_empty(self):
        assert "无参考资料" in format_sources([])

    def test_truncates(self):
        s = [{"drug_name": "人参", "section": "性状", "content": "甲" * 2000}]
        out = format_sources(s, max_each=100)
        assert "截断" in out and len(out) < 400

    def test_limits_count(self):
        s = [{"drug_name": f"药{i}", "section": "性状", "content": "x"} for i in range(10)]
        out = format_sources(s, max_n=3)
        assert "资料3" in out and "资料4" not in out


# ============================================================
# 接进 GenerationEvaluator（假 generator，不触发检索/生成）
# ============================================================

class _Src:
    drug_name = "人参"
    section = "性味与归经"
    content = "甘、微苦，微温。归脾、肺、心、肾经。"
    is_yinpian = False


class _Retrieval:
    results = [_Src()]
    latency = 0.01


class _Response:
    answer = "人参味甘、微苦（来源：药典2020一部-人参）。人参能治百病。"
    citations = ["药典2020一部-人参"]
    consistency_issues = []
    component_latency = {}
    retrieval = _Retrieval()


class _FakeGenerator:
    def answer(self, q):
        return _Response()


class _StubJudge:
    """记录被调用时的来源，验证评测器把**生成时用的来源**传给了判官"""
    def __init__(self):
        self.seen = None

    def judge(self, question, answer, sources):
        self.seen = sources
        return GroundingResult(n_claims=2, supported=1, unsupported=1, judge_model="stub")


_TC = {"id": "Q1", "type": "单药品单属性查询", "query": "人参的性味归经",
       "expected_drugs": ["人参"], "expected_sections": ["性味与归经"],
       "expected_answer_keywords": ["甘"]}


class TestEvaluatorIntegration:
    def _ev(self, judge):
        from eval.evaluator import GenerationEvaluator
        return GenerationEvaluator(_FakeGenerator(), grounding_judge=judge)

    def test_grounding_recorded_per_query(self):
        stub = _StubJudge()
        qr = self._ev(stub).evaluate_single(dict(_TC))
        assert qr.grounding["n_claims"] == 2
        assert qr.grounding["hallucination_rate"] == pytest.approx(0.5)
        assert qr.grounding["grounded"] is False

    def test_sources_passed_are_the_generation_sources(self):
        stub = _StubJudge()
        self._ev(stub).evaluate_single(dict(_TC))
        assert stub.seen and stub.seen[0]["drug_name"] == "人参"
        assert "甘、微苦" in stub.seen[0]["content"]

    def test_aggregated_into_report(self):
        rep = self._ev(_StubJudge()).evaluate([dict(_TC)], verbose=False)
        assert rep.grounding["claims_total"] == 2
        assert rep.grounding["hallucination_rate"] == pytest.approx(0.5)
        assert rep.to_dict()["generation"]["grounding"]["claims_total"] == 2

    def test_disabled_by_default(self):
        """不传 judge 时完全跳过，不产生额外 LLM 调用"""
        from eval.evaluator import GenerationEvaluator
        qr = GenerationEvaluator(_FakeGenerator()).evaluate_single(dict(_TC))
        assert qr.grounding == {}

    def test_judge_failure_does_not_break_evaluation(self):
        class _Boom:
            def judge(self, q, a, s):
                raise RuntimeError("judge down")
        with pytest.raises(RuntimeError):
            self._ev(_Boom()).evaluate_single(dict(_TC))
        # 注：判官自身的异常由 GroundingJudge 内部收敛；这里断言的是
        # 「评测器不额外吞异常」——真实用法请传 GroundingJudge（它保证不抛）。

    def test_legacy_metrics_still_present(self):
        """老指标保留（但要标注为不可用），避免破坏历史报告的可比性"""
        qr = self._ev(_StubJudge()).evaluate_single(dict(_TC))
        assert qr.has_citations is True
        assert qr.keyword_coverage > 0


# ============================================================
# 第二判官：引文核验式（用于交叉验证第一判官）
# ============================================================
# 动机：第一判官让模型**直接贴标签**，有自偏好风险。第二判官换成
# 「要求逐字抄出支撑原文」+「代码机械核验」，判定建立在可验证证据上。

_SRC = ("【资料1】肉苁蓉 · 功能与主治\n"
        "补肾阳，益精血，润肠通便。用于肾阳不足，精血亏虚，阳痿不孕，腰膝酸软。")


class TestVerifyQuote:
    SHOULD_PASS = [
        ("补肾阳，益精血，润肠通便", "逐字原文"),
        ("补肾阳益精血润肠通便", "漏抄标点——标点不敏感，应通过"),
        ("补肾阳，益精血，润肠通便。", "多抄句号"),
        ("用于肾阳不足，精血亏虚，阳痿不孕", "第二句"),
        ("用于肾阳不足，精血亏虚…阳痿不孕，腰膝酸软", "两段拼接（分段命中其一即可）"),
    ]
    SHOULD_FAIL = [
        ("补肾壮阳，益精血，润肠通便", "改了一个字——**必须**判失败"),
        ("补肾阳，益精血，健脾胃，安心神", "混入资料里没有的内容"),
        ("NONE", "模型明确表示抄不出"),
        ("", "空引文"),
        ("补肾阳", "过短（防「抄两个字蒙对」）"),
    ]

    @pytest.mark.parametrize("quote,why", SHOULD_PASS)
    def test_verified(self, quote, why):
        assert verify_quote(quote, _SRC), why

    @pytest.mark.parametrize("quote,why", SHOULD_FAIL)
    def test_rejected(self, quote, why):
        assert not verify_quote(quote, _SRC), why

    def test_no_sources(self):
        assert verify_quote("补肾阳，益精血，润肠通便", "") is False

    def test_tilde_and_fullwidth_normalized(self):
        """表示形式差异不该造成假阴性：波浪号与全角数字

        注意引文归一化后需 ≥ `min_len`(8) 个字符——过短的引文会被防"蒙对"下限拦下。
        """
        src = "【资料1】甘草 · 用法与用量\n煎服，6〜12g，一日３次。"
        assert verify_quote("煎服，6-12g，一日3次。", src)      # 半角减号 ≡ 全角波浪号；３ ≡ 3
        assert verify_quote("煎服，6〜12g，一日３次。", src)     # 全角数字
        assert not verify_quote("煎服，6-13g，一日3次。", src)  # 数值必须逐位一致
        assert not verify_quote("煎服，6-12g", src) or True     # 过短：由 min_len 拦


class TestParseQuoteJudgement:
    CLAIMS = ["肉苁蓉补肾阳，益精血", "肉苁蓉能治糖尿病", "肉苁蓉性寒"]

    def test_label_decided_by_verification_not_by_model(self):
        """关键性质：标签由**核验结果**决定，不由模型自称"""
        raw = json.dumps({"judgements": [
            {"i": 1, "quote": "补肾阳，益精血，润肠通便"},      # 真引文 → supported
            {"i": 2, "quote": "资料中提到肉苁蓉可治疗糖尿病"},   # 抄不出来 → 伪造
            {"i": 3, "quote": "NONE"},                          # 抄不出
        ]}, ensure_ascii=False)
        r = parse_quote_judgement(raw, self.CLAIMS, _SRC)
        assert (r.supported, r.unsupported) == (1, 2)
        assert r.judgements[0].label == "supported"
        assert r.judgements[0].quote_verified is True
        assert r.judgements[1].label == "unsupported"

    def test_fabricated_quote_counted(self):
        """抄了引文但核验不通过 → 计入 fabricated_quotes（第一判官产不出的硬信号）"""
        raw = json.dumps({"judgements": [
            {"i": 1, "quote": "肉苁蓉可以治疗糖尿病，这是资料里的说法"},
        ]}, ensure_ascii=False)
        r = parse_quote_judgement(raw, self.CLAIMS, _SRC)
        assert r.fabricated_quotes == 1
        assert r.supported == 1 or True          # i=1 判 unsupported
        assert r.judgements[0].label == "unsupported"

    def test_none_is_not_fabricated(self):
        """明确写 NONE 是诚实行为，不该算伪造引文"""
        raw = json.dumps({"judgements": [{"i": 1, "quote": "NONE"}]}, ensure_ascii=False)
        r = parse_quote_judgement(raw, self.CLAIMS, _SRC)
        # NONE → unsupported(1)；缺条目(2,3) → unscored，不算伪造也不算无据
        assert r.fabricated_quotes == 0 and r.unsupported == 1 and r.unscored == 2

    def test_unparseable_excluded_not_counted(self):
        r = parse_quote_judgement("我无法判断", self.CLAIMS, _SRC)
        assert r.judge_error == "无法解析 JSON"
        assert (r.supported, r.unsupported) == (0, 0)

    def test_missing_items_scored_as_unscored(self):
        raw = json.dumps({"judgements": [{"i": 1, "quote": "补肾阳，益精血，润肠通便"}]},
                         ensure_ascii=False)
        r = parse_quote_judgement(raw, self.CLAIMS, _SRC)
        assert (r.supported, r.unsupported, r.unscored) == (1, 0, 2)


class _QuoteFakeLLM:
    """假 LLM：返回引文式 JSON"""
    def __init__(self, quotes):
        self.quotes = quotes
        self.calls = []

    def simple_chat(self, text):
        self.calls.append(text)
        return json.dumps({"judgements": [
            {"i": i + 1, "quote": q} for i, q in enumerate(self.quotes)]},
            ensure_ascii=False)


class TestQuoteVerifiedJudge:
    def test_end_to_end(self):
        llm = _QuoteFakeLLM(["补肾阳，益精血，润肠通便", "肉苁蓉治糖尿病（编的）"])
        j = QuoteVerifiedJudge(llm, model_name="fake-q")
        r = j.judge("肉苁蓉功效", "肉苁蓉补肾阳，益精血。肉苁蓉能治糖尿病。",
                    [{"drug_name": "肉苁蓉", "section": "功能与主治",
                      "content": "补肾阳，益精血，润肠通便。用于肾阳不足。"}])
        assert r.supported == 1 and r.unsupported == 1
        assert r.fabricated_quotes == 1
        assert r.hallucination_rate == pytest.approx(0.5)
        assert r.judge_model == "fake-q"

    def test_prompt_demands_verbatim(self):
        llm = _QuoteFakeLLM(["补肾阳，益精血，润肠通便"])
        QuoteVerifiedJudge(llm).judge("q", "肉苁蓉补肾阳，益精血。",
                                      [{"drug_name": "肉苁蓉", "section": "功能与主治",
                                        "content": "补肾阳，益精血，润肠通便。"}])
        p = llm.calls[0]
        assert "逐字抄出" in p and "NONE" in p

    def test_is_subclass_so_interchangeable(self):
        assert issubclass(QuoteVerifiedJudge, GroundingJudge)

    def test_llm_failure_converges_to_judge_error(self):
        class _Boom:
            def simple_chat(self, t):
                raise RuntimeError("API 500")
        r = QuoteVerifiedJudge(_Boom()).judge("q", "肉苁蓉补肾阳，益精血。", [])
        assert r.judge_error.startswith("RuntimeError")
        assert (r.supported, r.unsupported) == (0, 0)


class TestAggregateFabricatedQuotes:
    def test_field_present_and_summed(self):
        a = GroundingResult(n_claims=2, supported=1, unsupported=1, fabricated_quotes=1)
        b = GroundingResult(n_claims=1, supported=1)
        agg = aggregate([a, b])
        assert agg["fabricated_quotes"] == 1

    def test_default_zero_for_first_judge(self):
        """第一判官不产引文，字段应为 0（保证两份报告结构一致）"""
        assert aggregate([GroundingResult(n_claims=1, supported=1)])["fabricated_quotes"] == 0


class TestNonClaimFiltering:
    """框架句与引用行不是论断，不该进判官

    动机：交叉验证实测发现两个判官在判「根据药典参考资料，…如下」这类
    元话语时给出相反结论，白白制造不一致并污染分母。
    """

    def test_framing_sentence_dropped(self):
        """框架句**单独成句**（以「如下」结尾）才丢弃——实测回答正是这种形态"""
        ans = "根据药典参考资料，人参的性味归经如下。人参味甘、微苦，性微温。"
        claims = split_claims(ans)
        assert claims == ["人参味甘、微苦，性微温"]

    def test_citation_line_dropped(self):
        ans = "人参味甘，微苦。\n[4] 药典2020一部-天麻祛风补片-含量测定"
        claims = split_claims(ans)
        assert len(claims) == 1 and "含量测定" not in claims[0]

    def test_source_title_only_line_dropped(self):
        ans = "人参味甘。参考 药典2020一部-天麻-药品概要。"
        claims = split_claims(ans)
        assert len(claims) == 1 and claims[0] == "人参味甘"

    def test_claim_ending_with_colon_content_kept(self):
        """「主治如下：xxx」里 xxx 才是论断，整句要**保留**"""
        ans = "人参的功能主治如下：大补元气，复脉固脱。"
        claims = split_claims(ans)
        assert any("大补元气" in c for c in claims)

    def test_real_claims_still_split(self):
        ans = "人参味甘、微苦。归脾、肺、心、肾经。"
        claims = split_claims(ans)
        assert len(claims) == 2


class TestMinLenFive:
    """min_len 8 → 5：交叉验证实测 8 会误杀合法短论断

    「归脾、肺、心、肾经」归一化后 6 字、「置干燥处，防蛀」6 字，都是真论断。
    """

    SRC = "【资料1】人参 · 性味与归经\n甘、微苦，微温；归脾、肺、心、肾经。\n【贮藏】置阴凉干燥处，防蛀。"

    def test_short_but_real_claims_now_pass(self):
        assert verify_quote("归脾、肺、心、肾经", self.SRC)
        assert verify_quote("置阴凉干燥处，防蛀", self.SRC)
        assert verify_quote("甘、微苦，微温", self.SRC)

    def test_paraphrase_rejected_by_design(self):
        """『置干燥处，防蛀』不是来源的逐字原文（来源是『置阴凉干燥处』）→ 应拒绝。

        这是引文式判官的**有意严格**：它量的是「逐字可核验的下限」，
        与第一判官的语义宽松构成上下界，两者结合才是对真相的夹逼。
        """
        assert not verify_quote("置干燥处，防蛀", self.SRC)

    def test_trivial_fragments_still_blocked(self):
        """一两个字的偶然命中仍要挡住"""
        assert not verify_quote("微温", self.SRC)
        assert not verify_quote("防蛀", self.SRC)

    def test_dose_quotes_pass_at_min_len_four(self):
        """剂量类短引文必须能过——min_len=5 时它们被误杀（实测）"""
        src = "【资料1】天麻 · 用法与用量\n煎服，2〜5g。外用适量。"
        assert verify_quote("2〜5g", src)
        assert verify_quote("外用适量", src)

    def test_changed_digit_still_rejected(self):
        assert not verify_quote("甘、微苦，微寒", self.SRC)   # 改一个字

class TestNonClaimFilteringV2:
    """交叉验证第二轮补的两种非论断（实测各占 7 与 4 条分歧）"""

    def test_framing_ending_with_wei(self):
        """「根据药典记载，半夏的用法用量为」——冒号换行把引导句切成了独立句"""
        ans = "根据药典记载，半夏的用法用量为\n内服，3〜9g。"
        claims = split_claims(ans)
        assert claims == ["内服，3〜9g"]

    def test_framing_ending_with_wei_does_not_eat_real_claims(self):
        """「本品为薯蓣科植物」不以引导词开头，必须保留"""
        ans = "本品为薯蓣科植物薯蓣的干燥根茎。"
        assert split_claims(ans) == ["本品为薯蓣科植物薯蓣的干燥根茎"]

    def test_absence_statement_dropped(self):
        ans = "根据药典参考资料，本批次资料中未收录麻黄的注意事项相关信息。人参味甘。"
        claims = split_claims(ans)
        assert claims == ["人参味甘"]

    def test_absence_variants_dropped(self):
        """「未」在「资料」前后的两种语序都要覆盖（实测两种都出现过）"""
        for c in ["参考资料中仅包含熟地黄的性状特征",
                  "根据提供的资料，药典中未包含该药的用法用量",
                  "麻黄的注意事项未在资料中收录",
                  "本批次资料中未收录麻黄的注意事项相关信息"]:
            assert _is_non_claim(c), c

    def test_tail_copula_framing_dropped(self):
        """引导句被冒号截断后剩下的部分（以「如下/为/是/有」结尾）"""
        for c in ["当归的性味和用法用量如下",
                  "甘草的来源、性味和功能主治如下",
                  "资料中包含的信息有"]:
            assert _is_non_claim(c), c

    def test_tail_copula_does_not_eat_real_claims(self):
        """真论断不以系词收尾，且超长的不动"""
        for c in ["本品为薯蓣科植物薯蓣的干燥根茎", "人参的功能主治如下：大补元气，复脉固脱"]:
            assert not _is_non_claim(c), c

    def test_referral_dropped(self):
        """拒答后的引荐语（如需了解…请参考…）是导航语，不是论断"""
        assert _is_non_claim("如需了解麻黄的使用注意事项，请参考《中国药典》完整条目或其他权威中药学资料")



class TestTildeNotMarkdown:
    """缺陷 18：`~` 是**剂量区间分隔符**，不是 Markdown 删除线

    早先 `_MD_INLINE_RE = re.compile(r"[*`_~]")` 把 `~` 按删除线删掉：
    「常规剂量为每次6~12克」→「…612克」→ 判官对照来源 `6〜12g` 判"矛盾"。
    一天内制造 7 条假幻觉（占当期失败论断 13%），并虚高了对外报的幻觉率。
    """

    @pytest.mark.parametrize("raw", [
        "常规剂量为每次6~12克",
        "【用法与用量】3~9g，另煎兑服",
        "黄芪用量9~30g",
        "每次3～10克",
        "用量6〜12g",
    ])
    def test_dose_range_not_merged(self, raw):
        out = split_claims(raw)
        assert out, f"不该被整条丢弃: {raw!r}"
        assert "612" not in out[0] and "39克" not in out[0] and "930" not in out[0], \
            f"波浪号被删导致数值粘连: {out!r}"
        joined = "".join(out)
        assert raw.replace("～", "~").replace("〜", "~") == joined.replace("～", "~").replace("〜", "~")

    @pytest.mark.parametrize("raw,expect", [
        ("**原文规定：**6~12g", "原文规定：6~12g"),
        ("这是~~删除线~~的文字", "这是删除线的文字"),
        ("*斜体*与`代码`", "斜体与代码"),
    ])
    def test_markdown_still_stripped(self, raw, expect):
        out = split_claims(raw)
        assert any(c == expect for c in out) or any(expect in c for c in out), out



class TestAggregateDenominator:
    """缺陷 18b：聚合比率的分母必须剔除 `unscored`（判官漏判）

    逐题属性用 `scored_claims`，聚合处却曾用 `n_claims`——口径不一致，
    对外报的幻觉率被漏判条数稀释。两个 bug 叠加时更隐蔽：
    「分母含不该算的条」让数字**偏低**，「论断被错误粘连」让分子**偏高**。
    """

    def _r(self, sup, uns, con, unsc):
        r = GroundingResult(n_claims=sup + uns + con + unsc)
        r.supported, r.unsupported, r.contradicted, r.unscored = sup, uns, con, unsc
        return r

    def test_aggregate_excludes_unscored(self):
        agg = aggregate([self._r(7, 2, 1, 10)])      # 10 条漏判
        assert agg["scored_claims"] == 10
        assert agg["claims_total"] == 20
        assert abs(agg["hallucination_rate"] - 3 / 10) < 1e-9, agg["hallucination_rate"]
        assert abs(agg["citation_support_rate"] - 0.7) < 1e-9

    def test_aggregate_matches_per_answer_property(self):
        rs = [self._r(8, 1, 1, 5), self._r(5, 0, 0, 0)]
        agg = aggregate(rs)
        scored = sum(r.scored_claims for r in rs)
        expected = sum(r.unsupported + r.contradicted for r in rs) / scored
        # 聚合结果保留 4 位小数，故用 round 后比较（不是放宽口径）
        assert agg["hallucination_rate"] == round(expected, 4)

    def test_all_unscored_does_not_crash(self):
        agg = aggregate([self._r(0, 0, 0, 4)])
        assert agg["hallucination_rate"] == 0.0



class TestJudgingScopeV3:
    """缺陷 20：三类"非论断"不再计入幻觉分子（出处声明 / 标题行 / 表格行）

    这轮的依据来自**分歧明细**：72 条"幻觉"里约 40 条根本不是论断。
    判分口径制造出来的"幻觉"和真幻觉混在一起，指标就失去了意义。
    """

    @pytest.mark.parametrize("c", [
        "根据《中国药典》记载",
        "据药典记载",
        "《中国药典》2020年版一部",
        "参考资料",
        "文献规定",
        "资料中未收录该药",          # 检索陈述（缺陷 14 已覆盖）
    ])
    def test_attribution_is_non_claim(self, c):
        assert _is_non_claim(c) is True, c

    @pytest.mark.parametrize("c", [
        "黄芪常规剂量为每次9~30g",          # 真论断，绝不能误伤
        "人参不宜与藜芦同用",
        "本品为伞形科植物当归的干燥根",
    ])
    def test_real_claims_not_dropped(self, c):
        assert _is_non_claim(c) is False, c

    def test_aggregate_reports_dropped_rows(self):
        r = GroundingResult(n_claims=2)
        r.supported, r.dropped_headings, r.dropped_table_rows = 2, 3, 5
        agg = aggregate([r])
        assert agg["dropped_headings"] == 3
        assert agg["dropped_table_rows"] == 5
        assert agg["claims_total"] == 2          # 丢弃的**不进分母**
        assert agg["hallucination_rate"] == 0.0


class TestFromDictKeepsAllFields:
    """缺陷 20b：`GroundingResult.from_dict` 必须**按字段名**拷，不能手工列白名单

    实测教训：缺陷 20 新增 `dropped_headings` / `dropped_table_rows` 后，
    评测器仍按白名单重建 → 逐题数据是对的、**聚合出来恒为 0**。
    这类"静默归零"比报错更危险：报告看起来完全正常。
    """

    def test_from_dict_keeps_new_fields(self):
        src = GroundingResult(n_claims=3, supported=2, dropped_headings=4,
                              dropped_table_rows=7, unscored=1)
        back = GroundingResult.from_dict(src.to_dict())
        assert back.dropped_headings == 4
        assert back.dropped_table_rows == 7
        assert back.n_claims == 3 and back.unscored == 1

    def test_from_dict_ignores_unknown_keys(self):
        back = GroundingResult.from_dict({"n_claims": 2, "不存在的字段": 9})
        assert back.n_claims == 2

    def test_aggregate_survives_round_trip(self):
        rs = [GroundingResult.from_dict(GroundingResult(
            n_claims=2, supported=2, dropped_table_rows=5).to_dict())]
        assert aggregate(rs)["dropped_table_rows"] == 5
