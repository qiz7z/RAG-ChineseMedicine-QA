# -*- coding: utf-8 -*-
"""
判分口径回归测试
================
覆盖 src/eval/evaluator.py 的两档判分口径（loose / strict），并守住一条
设计风险：两套引擎**刻意各写一份**判分实现（零共享 import），口径极易漂移，
TestEngineParity 直接对拍两份实现的判定结果。

另含历史基线回归（需要 data/ 与报告文件，缺失自动 skip）：
把当前判分代码应用回 7-06 / 8-30 两份历史报告，断言仍能复现
loose 91.0%/89.0% 与 strict 83.1%/82.0%。

运行：
    python -m pytest tests/test_eval_criteria.py -v
"""
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval.evaluator import (  # noqa: E402
    attribute_strict_miss, drug_match_loose, drug_match_strict,
    judge_result, section_match, normalize_text,
)


def _res(drug="人参", section="完整条目", content="", is_yinpian=False):
    """构造最小可判定的检索结果替身（字段与 SearchResult 对齐）"""
    return SimpleNamespace(drug_name=drug, section=section,
                           content=content, is_yinpian=is_yinpian)


# ============================================================
# loose 口径
# ============================================================

class TestLooseCriteria:
    def test_bidirectional_substring(self):
        assert drug_match_loose("人参-饮片", ["人参"]) is True
        assert drug_match_loose("人参", ["人参-饮片"]) is True
        assert drug_match_loose("黄芪", ["人参"]) is False

    def test_auto_hit_without_expected(self):
        # 历史口径：无期望药品 → 无条件命中（这正是虚高的来源之一，保留仅为可比）
        assert drug_match_loose("任意药品", []) is True
        assert drug_match_loose("", []) is True


# ============================================================
# strict 口径
# ============================================================

class TestStrictDrugCriteria:
    def test_exact_name_hits(self):
        assert drug_match_strict("人参", ["人参"]) is True

    def test_yinpian_entry_hits(self):
        # 该药品的饮片条目算命中
        assert drug_match_strict("人参-饮片", ["人参"], is_yinpian=True) is True

    def test_formulation_rejected(self):
        # 子串匹配的放大器：含该药的成方制剂不是该药材
        assert drug_match_strict("黄连胶囊", ["黄连"], is_yinpian=False) is False
        assert drug_match_strict("天麻祛风补片", ["天麻"], is_yinpian=False) is False
        assert drug_match_strict("复方丹参喷雾剂", ["丹参"], is_yinpian=False) is False

    def test_yinpian_flag_required(self):
        # 同一名称但没有饮片标记 → 不能按饮片规则放行
        assert drug_match_strict("人参-饮片", ["人参"], is_yinpian=False) is False

    def test_empty_expected_never_matches(self):
        assert drug_match_strict("人参", []) is False


class TestStrictSectionCriteria:
    def test_mark_in_content(self):
        content = "【性味与归经】甘、微苦，微温。\n【功能与主治】大补元气。"
        assert section_match(content, "完整条目", ["性味与归经"]) is True

    def test_section_field_equality(self):
        assert section_match("正文无标记", "性状", ["性状"]) is True

    def test_merged_bucket_not_confused(self):
        # ETL 把多个原生章节合并成 "临床应用" 桶，桶名不能当章节命中
        content = "【功能与主治】补气。"
        assert section_match(content, "临床应用", ["性味与归经"]) is False

    def test_source_special_case(self):
        # 【来源】正文在 summary / whole_entry 里不带标记
        assert section_match("概述：为菊科植物…", "药品概要", ["来源"]) is True
        assert section_match("正文无标记", "完整条目", ["来源"]) is True
        assert section_match("【性状】…", "性状", ["来源"]) is False

    def test_empty_inputs(self):
        assert section_match("", "完整条目", ["来源"]) is False   # 正文为空 → 无法验证
        assert section_match("正文", "完整条目", []) is False
        assert section_match("", "性状", ["性状"]) is False


class TestJudgeResult:
    def test_strict_pass(self):
        r = _res("人参", "完整条目", "【性味与归经】甘。")
        loose, strict, reason = judge_result(r, ["人参"], ["性味与归经"])
        assert (loose, strict, reason) == (True, True, "")

    def test_drug_not_exact(self):
        r = _res("黄连胶囊", "临床应用", "【功能与主治】…")
        loose, strict, reason = judge_result(r, ["黄连"], ["性状"])
        assert (loose, strict, reason) == (True, False, "drug_not_exact")

    def test_section_missing(self):
        r = _res("黄连", "性状", "【性状】本品为毛茛科植物…")
        loose, strict, reason = judge_result(r, ["黄连"], ["含量测定"])
        assert (loose, strict, reason) == (True, False, "section_missing")

    def test_unevaluable(self):
        r = _res("任意药品", "完整条目", "正文")
        loose, strict, reason = judge_result(r, [], [])
        assert (loose, strict, reason) == (True, False, "unevaluable")


class TestMissAttribution:
    def test_not_recalled(self):
        rs = [_res("黄芪"), _res("甘草")]
        assert attribute_strict_miss(rs, ["人参"]) == "drug_not_recalled"

    def test_not_exact(self):
        # 宽松能匹配（子串），但都不是该药材本身
        rs = [_res("人参口服液"), _res("人参茶")]
        assert attribute_strict_miss(rs, ["人参"]) == "drug_not_exact"

    def test_section_missing(self):
        rs = [_res("人参", "性状", "【性状】…")]
        assert attribute_strict_miss(rs, ["人参"], k=5) == "section_missing"

    def test_empty_drug_name_never_matches(self):
        # `"" in "人参"` 为真，必须显式挡住，否则空名结果白拿分
        assert drug_match_loose("", ["人参"]) is False
        assert attribute_strict_miss([_res("")], ["人参"]) == "drug_not_recalled"


# ============================================================
# 两套引擎判分实现一致性（对拍）
# ============================================================

def _load_langchain_eval():
    """按路径加载 langchain_app/eval.py（不污染 sys.modules / sys.path）"""
    lc_dir = ROOT / "langchain_app"
    if not (lc_dir / "eval.py").exists():
        pytest.skip("langchain_app/eval.py 不存在")
    saved_config = sys.modules.pop("config", None)
    sys.path.insert(0, str(lc_dir))
    try:
        spec = importlib.util.spec_from_file_location("_lc_eval_probe", lc_dir / "eval.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:                      # 缺 langchain 依赖等
        pytest.skip(f"无法加载 langchain_app/eval.py: {e}")
    finally:
        sys.path.remove(str(lc_dir))
        sys.modules.pop("config", None)
        if saved_config is not None:
            sys.modules["config"] = saved_config


class TestEngineParity:
    """手撕版与标准版的判分函数必须在同一 case 矩阵上给出完全一致的判定"""

    CASES = [
        # (drug_name, section, content, is_yinpian, expected_drugs, expected_sections)
        ("人参", "完整条目", "【性味与归经】甘。", False, ["人参"], ["性味与归经"]),
        ("人参-饮片", "完整条目", "【性状】…", True, ["人参"], ["性味与归经"]),
        ("人参-饮片", "完整条目", "【性状】…", False, ["人参"], ["性状"]),
        ("黄连胶囊", "临床应用", "【功能与主治】…", False, ["黄连"], ["性状"]),
        ("天麻祛风补片", "临床应用", "【性状】…", False, ["天麻"], ["性味与归经"]),
        ("甘草", "药品概要", "概述：为豆科植物甘草…", False, ["甘草"], ["来源"]),
        ("甘草", "完整条目", "本品为豆科植物…", False, ["甘草"], ["来源"]),
        ("黄芪", "性状", "【性状】圆柱形。", False, ["黄芪"], ["含量测定"]),
        ("任意药品", "完整条目", "正文", False, [], []),
        ("", "完整条目", "", False, ["人参"], ["性状"]),
        ("人参", "", "", False, ["人参"], []),
    ]

    def test_verdicts_match(self):
        lc = _load_langchain_eval()
        for drug, section, content, is_yin, ed, es in self.CASES:
            src_v = judge_result(_res(drug, section, content, is_yin), ed, es)
            lc_v = lc._judge(drug, section, content, is_yin, ed, es)
            assert src_v == lc_v, f"口径漂移: {(drug, section, ed, es)} src={src_v} lc={lc_v}"

    def test_langchain_loose_helper_kept(self):
        # 既有测试/脚本依赖的旧签名必须保留
        lc = _load_langchain_eval()
        assert lc._check_hit("人参-饮片", ["人参"]) is True
        assert lc._check_hit("任意药品", []) is True


class TestKeywordNormalization:
    """关键词覆盖率判分前的文本规范化。

    背景：药典原文用「〜」(U+301C) 写剂量范围，测试集 expected_answer_keywords
    却用半角「-」（期望 "6-12g"、原文 "6〜12g"）→ 答对的题被判 0 分。
    实测这一差异让关键词覆盖率系统性低估约 7~8pp（见 docs/08）。
    """

    DASH_CASES = ["6〜12g", "6～12g", "6~12g", "6–12g", "6—12g", "6－12g", "6-12g"]

    def test_dash_variants_folded(self):
        for c in self.DASH_CASES:
            assert normalize_text(c) == "6-12g", f"未折叠: {c!r}"

    def test_fullwidth_folded(self):
        assert normalize_text("１０〜１５克") == "10-15克"
        assert normalize_text("ＡＢ") == "ab"

    def test_whitespace_removed(self):
        assert normalize_text("6〜12 g") == "6-12g"

    def test_empty(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""

    def test_matches_langchain_impl(self):
        """两套刻意重复的实现必须逐字对齐，否则口径会漂移"""
        lc = _load_langchain_eval()
        for c in self.DASH_CASES + ["１０〜１５克", "Ａ〜Ｂ", "100%", ""]:
            assert normalize_text(c) == lc.normalize_text(c), f"口径漂移: {c!r}"


# ============================================================
# 历史基线回归（数据缺失自动 skip）
# ============================================================

def _build_chunk_index():
    """历史基线重算用的 chunk 正文索引（**固化快照**，不随 ETL 演进变化）。

    历史报告的 `top_k_details` 只存了 (drug_name, section, preview)，而重算判分
    需要取回 chunk 的**完整正文**（判章节要在正文里找【】标记）。这里用
    `tests/fixtures/baseline_chunks.json` 固化所需 chunk，而不是读当前的
    `data/processed/chunks.json`——否则一旦重跑 ETL（chunk 边界与正文变化），
    历史基线就无法复现，测试会退化为"因数据变化而失败"，失去防口径漂移的意义。
    """
    p = ROOT / "tests" / "fixtures" / "baseline_chunks.json"
    if not p.exists():
        pytest.skip("历史基线 fixture 不存在（tests/fixtures/baseline_chunks.json）")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {
        tuple(key.split("\u0001")): {"content": v["content"], "is_yinpian": v["is_yinpian"]}
        for key, v in raw.items()
    }


def _recompute(report_path, idx):
    """把当前判分代码应用回一份历史报告，返回 (loose 命中, strict 命中, 分母)"""
    if not report_path.exists():
        pytest.skip(f"{report_path.name} 不存在")
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    qs = {q["id"]: q for q in json.loads(
        (ROOT / "data/eval/test_queries.json").read_text(encoding="utf-8"))["queries"]}

    loose_hits = strict_hits = evaluable = scored = 0
    for r in rep["results"]:
        q = qs[r["query_id"]]
        # 与评测器同口径：out_of_scope 题两项都不计分
        if q.get("out_of_scope"):
            continue
        scored += 1
        ed = q.get("expected_drugs") or []
        es = q.get("expected_sections") or []
        loose_rank = strict_rank = 0
        for i, d in enumerate(r.get("top_k_details") or [], 1):
            meta = idx.get((d.get("drug_name") or "", d.get("section") or "",
                            re.sub(r"\s+", " ", (d.get("content_preview") or "")).strip())) or {}
            shim = _res(d.get("drug_name") or "", d.get("section") or "",
                        meta.get("content") or "", bool(meta.get("is_yinpian")))
            lo, st, _ = judge_result(shim, ed, es)
            if not ed:
                if loose_rank == 0:
                    loose_rank = i
                continue
            if lo and loose_rank == 0:
                loose_rank = i
            if st and strict_rank == 0:
                strict_rank = i
        if 0 < loose_rank <= 5:
            loose_hits += 1
        if ed:
            evaluable += 1
            if 0 < strict_rank <= 5:
                strict_hits += 1
    return loose_hits, strict_hits, evaluable, scored


class TestHistoricalBaseline:
    """判分代码必须能复现已验证过的历史口径数字（防止口径被无意改宽/改窄）"""

    # ⚠️ 2026-09-22 口径收口后，**loose 的分母也从 100 变成 89**
    #（11 道 out_of_scope 题不再被 loose 无条件送分）。
    # 原来的 loose 91.0% / 89.0% 是"含 11 道白送分"的数字；扣掉后为 89.9% / 87.6%。
    # strict 本来就是 89 为分母，数值不变。
    @pytest.mark.parametrize("name,loose,strict", [
        # 上列数字是**收口口径下的重算值**（实测，非按比例推算）：
        #   7-06: loose 80/89、strict 74/89 ；8-30: loose 79/89、strict 73/89
        ("eval_retrieval_20260706_193011.json", 0.8989, 0.8315),   # 开重排
        ("eval_retrieval_20260830_103446.json", 0.8876, 0.8202),   # 关重排（上线配置）
    ])
    def test_rates(self, name, loose, strict):
        idx = _build_chunk_index()
        loose_hits, strict_hits, evaluable, scored = _recompute(
            ROOT / "data/eval/reports" / name, idx)
        assert scored == 89, "计分题数应为 89（100 题中 11 道 out_of_scope 不计分）"
        assert evaluable == 89, "可评测题数应为 89"
        assert loose_hits / scored == pytest.approx(loose, abs=0.001)
        assert strict_hits / evaluable == pytest.approx(strict, abs=0.001)


# ============================================================
# 测试集自身的口径不变量（2026-09-22 收口）
# ============================================================

def _load_test_set():
    p = ROOT / "data" / "eval" / "test_queries.json"
    if not p.exists():
        pytest.skip("测试集不存在")
    return json.loads(p.read_text(encoding="utf-8"))["queries"]


class TestTestSetScope:
    """守住测试集口径：计分题必须**全部可评测**，两档分母才能一致

    背景：原先有 11 道 `expected_drugs` 为空的题（10 道"方法通则查询"问的是药典**四部**
    通则正文，而语料只有**一部**；另 1 道是集合型横向查询）。它们在 loose 里被**无条件送分**
    （白送 11pp），在 strict 里被排除 → 两档分母不一致（loose 120 / strict 109）。
    现给它们打 `out_of_scope` 标记，由两套评测器统一剔除。
    """

    def test_all_scored_queries_are_evaluable(self):
        qs = _load_test_set()
        not_evaluable = [q["id"] for q in qs
                         if not q.get("out_of_scope") and not (q.get("expected_drugs") or [])]
        assert not not_evaluable, (
            f"计分题里仍有不可评测的题 {not_evaluable}——"
            f"会给它们补上期望药品，或标 out_of_scope（否则 loose 会白送分、两档分母不一致）")

    def test_out_of_scope_queries_have_reason(self):
        for q in _load_test_set():
            if q.get("out_of_scope"):
                assert q.get("out_of_scope_reason"), f'{q["id"]} 标了 out_of_scope 却没写原因'
                assert not (q.get("expected_drugs") or []), \
                    f'{q["id"]} 既有期望药品又标 out_of_scope，自相矛盾'

    def test_four_part_method_questions_are_out_of_scope(self):
        """四部通则题必须标 out_of_scope——一部语料里没有通则正文，无法评测"""
        qs = {q["id"]: q for q in _load_test_set()}
        for i in range(76, 86):
            qid = f"Q{i:03d}"
            if qid in qs:
                assert qs[qid].get("out_of_scope"), f"{qid} 是四部通则题，应标 out_of_scope"

    def test_ids_unique(self):
        ids = [q["id"] for q in _load_test_set()]
        assert len(ids) == len(set(ids))


class TestOutOfScopeParity:
    """两套判分实现刻意各写一份，`out_of_scope` 支持不能只落在一侧"""

    def test_both_reports_expose_the_field(self):
        from eval.evaluator import EvalReport as SrcReport
        assert "out_of_scope_queries" in SrcReport.__dataclass_fields__

        lc = _load_langchain_eval()
        if lc is None:
            pytest.skip("langchain_app/eval.py 不可导入")
        assert "out_of_scope_queries" in lc.EvalReport.__dataclass_fields__

    def test_both_evaluators_filter_it(self):
        """源码级断言：两处都必须有 out_of_scope 过滤（防止只改一侧）"""
        src = (ROOT / "src" / "eval" / "evaluator.py").read_text(encoding="utf-8")
        lc = (ROOT / "langchain_app" / "eval.py").read_text(encoding="utf-8")
        for name, text in (("src/eval/evaluator.py", src), ("langchain_app/eval.py", lc)):
            assert 'tc.get("out_of_scope")' in text, f"{name} 未过滤 out_of_scope"
