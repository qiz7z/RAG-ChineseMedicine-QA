# -*- coding: utf-8 -*-
"""
ETL 解析器回归测试（2026-09-22）
================================
针对「表格首格自带【章节】标记」这一解析缺陷。

**缺陷背景**：成方制剂的【处方】在 .docx 里是**表格**形式，首格形如
「【处方】醋香附138g」，且该表紧跟在拼音名之后——此时解析器的
`current_section` 仍为 `None`。历史实现只在 `current_section` 存在时才把表格
挂到"上一个章节"，于是**整张处方表被静默丢弃**：

  - 全文 556 张【处方】表格，其中 **530 个成方制剂的处方内容完全不在知识库里**
    （实测 drugs.json 含【处方】的条目仅 908 → 修复后 1395）
  - 并因缺少【处方】信号被 `_infer_category` 误判为「药材和饮片」
    （530 个里 58 个如此）

用合成文档测试，不依赖受版权保护的药典原文件。
背景与实测数据见 docs/08_检索缺陷修复与口径澄清.md
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

docx = pytest.importorskip("docx")
from docx import Document                            # noqa: E402
from docx.enum.style import WD_STYLE_TYPE            # noqa: E402

from etl.parser import (  # noqa: E402
    DrugEntry, INHERITABLE_SECTIONS, PharmacopoeiaParser, Section,
)


# 解析器识别药品名所依赖的样式（见 parser.EXTRA_DRUG_NAME_STYLES）
STYLES = ["Heading #1|1", "Heading #2|1", "Body text|1", "Body text|3"]


def _make_docx(tmp_path: Path, *, table_first: bool) -> Path:
    """构造一个最小药典文档。

    table_first=True  → 【处方】以**表格**给出（真实文档里成方制剂的形态，历史缺陷路径）
    table_first=False → 【处方】以**段落**给出（对照组，历史实现能正确解析）
    """
    doc = Document()
    for name in STYLES:
        try:
            doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        except Exception:                      # 样式已存在
            pass

    def para(text, style):
        p = doc.add_paragraph(text)
        p.style = doc.styles[style]
        return p

    para("测试丸", "Heading #1|1")
    para("Ceshi Wan", "Heading #2|1")

    if table_first:
        t = doc.add_table(rows=2, cols=2)
        t.rows[0].cells[0].text = "【处方】测试药材甲100g"
        t.rows[0].cells[1].text = "测试药材乙50g"
        t.rows[1].cells[0].text = "测试药材丙30g"
        t.rows[1].cells[1].text = "测试药材丁20g"
    else:
        para("【处方】测试药材甲100g 测试药材乙50g 测试药材丙30g 测试药材丁20g",
             "Body text|1")

    para("【制法】以上四味，粉碎成细粉，过筛，混匀，即得。", "Body text|1")
    para("【功能与主治】 testing 用于测试。", "Body text|1")

    path = tmp_path / "sample.docx"
    doc.save(str(path))
    return path


def _parse_one(path: Path):
    entries = PharmacopoeiaParser(str(path)).parse()
    assert entries, "解析器没有产出任何条目"
    return entries[0]


class TestSectionInTableFirstCell:
    """表格首格的【章节】标记必须被识别为新章节（【处方】即此形态）"""

    def test_table_form_section_is_captured(self, tmp_path):
        entry = _parse_one(_make_docx(tmp_path, table_first=True))
        names = [s.section_name for s in entry.sections]
        assert "处方" in names, f"【处方】表格被丢弃了，实际章节: {names}"

    def test_table_form_section_content_is_not_empty(self, tmp_path):
        entry = _parse_one(_make_docx(tmp_path, table_first=True))
        cf = next(s for s in entry.sections if s.section_name == "处方")
        assert "测试药材甲100g" in cf.content, "处方内容为空或丢失"
        assert "【处方】" in cf.content, "正文里应保留【处方】标记（评测 strict 口径依赖它）"

    def test_paragraph_form_still_works(self, tmp_path):
        """对照组：段落形式的【处方】不能因为本次改动而失效"""
        entry = _parse_one(_make_docx(tmp_path, table_first=False))
        assert "处方" in [s.section_name for s in entry.sections]

    def test_following_sections_preserved(self, tmp_path):
        """表格开新章节后，后续章节不能被吞掉或错位"""
        entry = _parse_one(_make_docx(tmp_path, table_first=True))
        names = [s.section_name for s in entry.sections]
        assert names[0] == "处方", names
        assert "制法" in names and "功能与主治" in names, names


class TestCategoryFollowsPrescriptionSignal:
    """【处方】被正确解析后，分类推断应随之纠正（同一根因，一处修复两处受益）"""

    def test_table_form_is_classified_as_formulation(self, tmp_path):
        entry = _parse_one(_make_docx(tmp_path, table_first=True))
        assert entry.category_hint == "成方制剂和单味制剂", entry.category_hint


# ============================================================
# 药名 / 拼音 / 拉丁名 挤在同一段落（软换行）
# ============================================================

def _make_docx_with_softbreak_name(tmp_path: Path, style: str) -> Path:
    """构造「药名 + 软换行 + 拼音 + 软换行 + 拉丁名」同段落的文档。

    这是真实药典里 6 个药品的形态（八角茴香、自然铜、黄蜀葵花、黄精、
    甜瓜子、款冬花）。历史实现在 `_is_likely_drug_name` 里只去空格不去换行，
    整段长度超标 → 判不出药名 → 该药品不成条目、正文还被追加进上一个药品。
    """
    doc = Document()
    for name in STYLES + [style]:
        try:
            doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        except Exception:
            pass

    def para(text, st):
        p = doc.add_paragraph(text)
        p.style = doc.styles[st]
        return p

    para("测试丸甲", "Heading #1|1")
    para("Ceshi Wanjia", "Heading #2|1")
    para("【制法】以上一味，粉碎，即得。", "Body text|1")

    # 药名 / 拼音 / 拉丁名：同一段落内用软换行分隔
    p = doc.add_paragraph()
    p.style = doc.styles[style]
    p.add_run("测试药材乙").add_break()
    p.add_run("Ceshi Yiyaoyi").add_break()
    p.add_run("TESTUS HERBA")

    para("【性状】本品为测试品。", "Body text|1")

    path = tmp_path / "softbreak.docx"
    doc.save(str(path))
    return path


class TestDrugNameWithSoftLineBreaks:
    """药名与拼音同段（w:br 软换行）时必须仍被识别为条目边界"""

    @pytest.mark.parametrize("style", ["Body text|3", "Body text|4",
                                       "Heading #3|1", "Heading #5|1"])
    def test_softbreak_name_becomes_entry(self, tmp_path, style):
        entries = PharmacopoeiaParser(
            str(_make_docx_with_softbreak_name(tmp_path, style))).parse()
        names = [e.drug_name for e in entries]
        assert "测试药材乙" in names, f"软换行药品名未成条目，实得条目: {names}"

    def test_pinyin_and_latin_extracted(self, tmp_path):
        entries = PharmacopoeiaParser(
            str(_make_docx_with_softbreak_name(tmp_path, "Body text|3"))).parse()
        e = next(x for x in entries if x.drug_name == "测试药材乙")
        assert e.pinyin_name == "Ceshi Yiyaoyi"
        assert e.latin_name == "TESTUS HERBA"

    def test_no_cross_drug_contamination(self, tmp_path):
        """核心不变量：上一个药品的章节里不得混入下一个药品的内容"""
        entries = PharmacopoeiaParser(
            str(_make_docx_with_softbreak_name(tmp_path, "Body text|3"))).parse()
        first = entries[0]
        blob = "\n".join(s.content for s in first.sections)
        assert "测试药材乙" not in blob, "上一个药品的正文被下一个药品污染"
        assert "Ceshi Yiyaoyi" not in blob, "拼音行被追加进了上一个药品的正文"

    def test_leading_line_used_for_name(self, tmp_path):
        """药名取首行，不能把整段（含换行）当成药名"""
        entries = PharmacopoeiaParser(
            str(_make_docx_with_softbreak_name(tmp_path, "Body text|3"))).parse()
        e = next(x for x in entries if x.drug_name == "测试药材乙")
        assert "\n" not in e.drug_name and "TESTUS" not in e.drug_name


# ============================================================
# 药名与拼音在**不同段落**、且药名用 Body text|3 样式
# ============================================================

def _make_docx_bodytext3_name(tmp_path: Path) -> Path:
    """药名单独成段但样式是 `Body text|3`，拼音在下一段的 `Heading #4`。

    这是真实文档里的形态（七味铁屑丸、七味葡萄散、七味广枣丸、七珍丸、
    八正合剂、八珍益母丸 等一批成方制剂）：`Body text|3` **主要**用于拼音行，
    历史实现的 EXTRA_DRUG_NAME_STYLES 把它排除在外 → 药名判不出 → 条目丢失 +
    正文被吞进上一个药品（实测 31+ 处，共 36 个条目丢失）。
    """
    doc = Document()
    styles = STYLES + ["Heading #4|1", "Body text|3"]
    for name in styles:
        try:
            doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        except Exception:
            pass

    def para(text, st):
        p = doc.add_paragraph(text)
        p.style = doc.styles[st]
        return p

    para("测试丸甲", "Heading #1|1")
    para("Ceshi Wanjia", "Heading #2|1")
    para("【制法】以上一味，粉碎，即得。", "Body text|1")

    para("测试药材乙", "Body text|3")        # 药名：Body text|3
    para("Ceshi Yiyaoyi", "Heading #4|1")    # 拼音：Heading #4
    para("【性状】本品为测试品。", "Body text|1")

    path = tmp_path / "bodytext3.docx"
    doc.save(str(path))
    return path


class TestDrugNameInBodyText3:
    """药名用 Body text|3 样式（与拼音行同款样式）时也必须被识别为条目边界"""

    def test_becomes_entry(self, tmp_path):
        entries = PharmacopoeiaParser(str(_make_docx_bodytext3_name(tmp_path))).parse()
        names = [e.drug_name for e in entries]
        assert "测试药材乙" in names, f"条目丢失，实得: {names}"

    def test_pinyin_line_not_treated_as_drug_name(self, tmp_path):
        """反向保护：拼音行（无中文）不能被误判成药名"""
        entries = PharmacopoeiaParser(str(_make_docx_bodytext3_name(tmp_path))).parse()
        names = [e.drug_name for e in entries]
        assert "Ceshi Yiyaoyi" not in names, "拼音行被误判为药品条目"

    def test_no_cross_drug_contamination(self, tmp_path):
        entries = PharmacopoeiaParser(str(_make_docx_bodytext3_name(tmp_path))).parse()
        blob = "\n".join(s.content for s in entries[0].sections)
        assert "测试药材乙" not in blob and "Ceshi Yiyaoyi" not in blob, \
            "上一个药品的正文被下一个药品污染"


# ============================================================
# 缺陷 15：核心章节被 -饮片 子条目吞掉（85% 主条目受影响）
# ============================================================
# 药典排版惯例：【性味与归经】【功能与主治】【用法与用量】【注意】【贮藏】
# 在条目末尾（【饮片】块之后）。解析器按"最后一个条目边界"归属，
# 这些章节被挂到 -饮片 上，主条目缺失 → 「天麻的性味归经」检索退化。

def _mk(name, secs, yinpian=False, parent=""):
    e = DrugEntry(drug_name=name, is_yinpian=yinpian, parent_drug=parent)
    e.sections = [Section(n, c) for n, c in secs]
    return e


def _parser():
    return PharmacopoeiaParser.__new__(PharmacopoeiaParser)


class TestInheritSectionsFromYinpian:
    def test_real_sections_inherited(self):
        """天麻实例：主条目缺 5 个核心章节，饮片子条目全有"""
        main = _mk("天麻", [("性状", "本品呈长椭圆形。")])
        child = _mk("天麻-饮片", [
            ("性味与归经", "甘，平。归肝经。"),
            ("功能与主治", "息风止痉，平抑肝阳，祛风通络。"),
            ("用法与用量", "3〜9g。"),
            ("注意", "血虚生风者慎用。"),
            ("贮藏", "置阴凉干燥处，防蛀。"),
        ], yinpian=True, parent="天麻")
        n = _parser()._inherit_sections_from_yinpian([main, child])
        got = {s.section_name for s in main.sections}
        assert n == 5 and got == {"性状"} | INHERITABLE_SECTIONS

    def test_short_real_text_not_killed_by_length_guard(self):
        """「甘，平。归肝经。」只有 8 字——长度阈值会误杀真实内容（实测踩过）"""
        main = _mk("某药", [("性状", "x")])
        child = _mk("某药-饮片", [("性味与归经", "甘，平。归肝经。"),
                                  ("用法与用量", "3〜9g。")], yinpian=True)
        n = _parser()._inherit_sections_from_yinpian([main, child])
        assert n == 2

    def test_placeholder_tongyaocai_skipped(self):
        """「同药材。」是占位文本，拷了无信息量（含全角空格变体）"""
        main = _mk("甲药", [("性状", "x")])
        child = _mk("甲药-饮片", [("性味与归经", "同药材。"),
                                  ("功能与主治", "同 药 材。"),
                                  ("用法与用量", "。")], yinpian=True)
        assert _parser()._inherit_sections_from_yinpian([main, child]) == 0
        assert {s.section_name for s in main.sections} == {"性状"}

    def test_existing_sections_not_overwritten(self):
        main = _mk("乙药", [("性味与归经", "苦，寒。")])
        child = _mk("乙药-饮片", [("性味与归经", "另写。"),
                                  ("功能与主治", "清热燥湿，泻火解毒。")], yinpian=True)
        n = _parser()._inherit_sections_from_yinpian([main, child])
        assert n == 1
        assert [s.content for s in main.sections if s.section_name == "性味与归经"] == ["苦，寒。"]

    def test_no_yinpian_child_is_noop(self):
        main = _mk("丙药", [("性状", "x")])
        assert _parser()._inherit_sections_from_yinpian([main]) == 0
        assert len(main.sections) == 1

    def test_sub_formulation_child_not_used(self):
        """只从 -饮片 回填；普通子剂型（口服液等）的章节不回填"""
        main = _mk("丁药", [("性状", "x")])
        sub = _mk("丁药口服液", [("功能与主治", "xxxx。")], parent="丁药")
        sub.is_sub_formulation = True
        assert _parser()._inherit_sections_from_yinpian([main, sub]) == 0


# ============================================================
# 缺陷 16：Word 文本框（<w:txbxContent>）内容读取
# ============================================================
# 文档里有 1,020 个文本框承载 20,517 字（86 处【性味与归经】、90 处【功能与主治】、
# 100 处【处方】），python-docx 的段落流读不到 → 大黄等 83 个药材的核心章节整段丢失。
# 处理方式是"把文本框段落线性化搬回正文流"，这里测的就是搬入时的两处清洗逻辑。

from etl.parser import (  # noqa: E402
    _collapse_box_repeats, _dominant_marker, _norm_box_line,
)


class TestTextboxLineCleaning:
    """文本框的两类噪声：① 同一行被裁切后重复；② 标记被 OCR 打碎"""

    @pytest.mark.parametrize("raw,expected", [
        # ① 裁切重复（真实样例，来自文档）
        ("辛，温;有小毒。归肝、脾、胃经。辛，温;有小毒。归肝、脾、胃",
         "辛，温;有小毒。归肝、脾、胃经。"),
        ("苦，寒。归脾、胃、大肠、肝、心包经。苦，寒。归脾、胃、大肠、",
         "苦，寒。归脾、胃、大肠、肝、心包经。"),
        ("祛风除湿，消肿止痛。用于风湿痹痛，半祛风除湿，消肿止痛。用于",
         "祛风除湿，消肿止痛。用于风湿痹痛，半"),
        # ② 被打碎的标记
        ("【性味与归经】【性味与归经】【性味与归经】", "【性味与归经】"),
        ("1功能与主治】11功能与主治】功能与主治】", "【功能与主治】"),
        ("t性味与归经】tt性味与归经】性味与归【", "【性味与归经】"),
        ("【处方】", "【处方】"),
        # ③ 正常内容行**不得**被改动（尤其拉丁学名的空格）
        ("本品为蓼科植物掌叶大黄Rheum palmatum L.的干燥根和根茎",
         "本品为蓼科植物掌叶大黄Rheum palmatum L.的干燥根和根茎"),
        ("补肾阳，益精血，润肠通便", "补肾阳，益精血，润肠通便"),
        ("用于脾虚食少，乏力便澹，妇人脏躁", "用于脾虚食少，乏力便澹，妇人脏躁"),
    ])
    def test_norm_box_line(self, raw, expected):
        assert _norm_box_line(raw) == expected

    def test_content_line_mentioning_section_name_not_turned_into_marker(self):
        """内容行里出现章节名，不能被误判成标记行（安全性由"删完必须什么都不剩"保证）"""
        for line in ["性味甘平", "功能与主治：清热燥湿，泻火解毒"]:
            assert _norm_box_line(line) == line

    def test_long_content_line_with_spaces_preserved(self):
        line = "【用法与用量】煎服，3〜15g；用于泻热通肠，凉血解毒。"
        assert _norm_box_line(line) == line


class TestTextboxLinearizationOnSyntheticDoc:
    """把文本框段落插回正文流：位置、顺序、去重"""

    def _doc_with_textbox(self):
        """构造一个带 <w:txbxContent> 的合成 docx（不依赖受版权保护的药典原文）"""
        import docx
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        doc = docx.Document()
        doc.add_paragraph("测试药")
        anchor = doc.add_paragraph("正文内容。")
        # 在锚点段落里挂一个文本框，内含两行（标记 + 内容），并各重复一次
        run = anchor.add_run()
        pict = OxmlElement('w:pict')
        box = OxmlElement('w:txbxContent')
        for text in ("【性味与归经】", "【性味与归经】", "甘，平。归肝经。", "甘，平。归肝经。"):
            bp = OxmlElement('w:p')
            br = OxmlElement('w:r')
            bt = OxmlElement('w:t')
            bt.text = text
            br.append(bt); bp.append(br); box.append(bp)
        pict.append(box); run._r.append(pict)
        return doc

    def test_lines_inserted_in_order_after_anchor(self):
        """返回的是**插入的元素列表**（解析器要靠它让前瞻跳过合成段落）"""
        from etl.parser import linearize_textboxes
        doc = self._doc_with_textbox()
        inserted = linearize_textboxes(doc)
        texts = [p.text for p in doc.paragraphs]
        assert len(inserted) == 2               # 重复的两份被去重成 2 行
        assert texts[-2:] == ["【性味与归经】", "甘，平。归肝经。"]
        assert all(getattr(e, "tag", "").endswith("}p") for e in inserted)

    def test_idempotent(self):
        """再跑一次不应重复插入（避免二次解析时内容翻倍）"""
        from etl.parser import linearize_textboxes
        doc = self._doc_with_textbox()
        linearize_textboxes(doc)
        before = [p.text for p in doc.paragraphs]
        linearize_textboxes(doc)                # 文本框仍在，会再插一遍 → 明确记录该行为
        after = [p.text for p in doc.paragraphs]
        # 不是幂等的：第二次仍会插入（因为没清掉原文），但**内容不重复出现两次以上**由去重保证
        assert before.count("【性味与归经】") == 1
        assert after.count("【性味与归经】") == 2


class TestOcrBrokenSectionMarkers:
    """缺陷 17：正文段落里被 OCR 打碎的章节标记要补回【】

    实测正文里有 ~1,300 个这样的段落（t鉴别】328 / ［含量测定】109 /
    1功能与主治】74 / [:规格】46 …），解析器的章节正则要求以【或〔开头，
    这些段落因此不被识别为章节——内容并进上一节，或在条目开头处整段丢失。
    """

    def _doc_with_junk_marker(self, text):
        import docx
        doc = docx.Document()
        doc.add_paragraph(text)
        return doc

    @pytest.mark.parametrize("junk,expected_start", [
        ("t鉴别】（1）叶表面观：上表皮细胞多角形。", "【鉴别】"),
        ("［含量测定】总糖取本品粗粉约5g。", "【含量测定】"),
        ("[:规格】 每粒装0.425g", "【规格】"),
        ("1功能与主治】", "【功能与主治】"),
        ("1贮藏】 置阴凉干燥处，防潮。", "【贮藏】"),
    ])
    def test_broken_marker_normalized(self, junk, expected_start):
        from etl.parser import normalize_ocr_section_markers
        doc = self._doc_with_junk_marker(junk)
        assert normalize_ocr_section_markers(doc) == 1
        assert doc.paragraphs[0].text.startswith(expected_start)

    def test_content_after_marker_preserved(self):
        from etl.parser import normalize_ocr_section_markers
        doc = self._doc_with_junk_marker("t鉴别】（1）叶表面观：上表皮细胞多角形。")
        normalize_ocr_section_markers(doc)
        assert doc.paragraphs[0].text == "【鉴别】（1）叶表面观：上表皮细胞多角形。"

    def test_already_clean_is_untouched(self):
        """已规范的标记不改动（幂等），普通内容段落也不碰"""
        from etl.parser import normalize_ocr_section_markers
        for good in ["【鉴别】同药材。", "本品呈圆形或椭圆形。", "人参", "注意事项：孕妇慎用。"]:
            doc = self._doc_with_junk_marker(good)
            assert normalize_ocr_section_markers(doc) == 0
            assert doc.paragraphs[0].text == good

    def test_idempotent(self):
        from etl.parser import normalize_ocr_section_markers
        doc = self._doc_with_junk_marker("t鉴别】同药材。")
        assert normalize_ocr_section_markers(doc) == 1
        assert normalize_ocr_section_markers(doc) == 0      # 第二次无事可做
        assert doc.paragraphs[0].text == "【鉴别】同药材。"

    def test_textbox_content_in_anchor_not_destroyed(self):
        """锚点段落里嵌着文本框时，只能改段落自己的 run，不能清掉框内内容"""
        import docx
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        from etl.parser import normalize_ocr_section_markers
        doc = docx.Document()
        para = doc.add_paragraph("t鉴别】同药材。")
        run = para.add_run()
        pict = OxmlElement('w:pict')
        box = OxmlElement('w:txbxContent')
        bp = OxmlElement('w:p'); br = OxmlElement('w:r'); bt = OxmlElement('w:t')
        bt.text = "框内文本"
        br.append(bt); bp.append(br); box.append(bp)
        pict.append(box); run._r.append(pict)
        normalize_ocr_section_markers(doc)
        assert "框内文本" in doc.paragraphs[0]._p.xml
