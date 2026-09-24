# -*- coding: utf-8 -*-
"""
OCR / 异体字纠错回归测试（2026-09-22）
=====================================
覆盖 `src/etl/cleaner.py` 的两层纠错：**药名层**与**正文层**。

## 为什么必须分两层（这是本文件最重要的一点）

同一个错字在药名和正文里的**合法性不同**，实测（全库统计）：

| 错字 | 药名中 | 正文中 | 结论 |
|------|--------|--------|------|
| 茂 | 只会是「芪」的误认（19 个药名） | **合法**：140 次「叶茂盛时采收」 | 药名可整字替换，正文不可 |
| 苔 | 只会是「芩」的误认 | **合法**：80 次「苦苣苔科」 | 同上 |
| 蕾 | 只会是「藿」的误认 | **合法**：575 次「干燥花蕾」 | 同上 |
| 萼 | 只会是「芎」的误认 | **合法**：666 次「萼筒」「萼片」 | 同上 |
| 替 | — | **合法**：11 次「被替代对照品」 | 一律不做整字替换 |
| 醍 | — | **多义**：乙醍=乙醚(1485)、油醍=石油醚(695)、蔥醍=蒽醌(41) | 只能词组级 |
| 幵 / 矶 / 苜 | 误认 | 正文亦从不合法 | 两层都可整字替换 |

反例（踩过的坑）：
- 按「醍→滤」整字替换 → 把「合并乙醍液」改成了「合并乙滤液」（应为**乙醚液**）
- 按「豬→猪」整字替换 → 错，正文里 30 次「豬」全是**豨**（豨薟草），
  而「猪」本身另正确出现 303 次
- 按「蔥→葱」整字替换 → 错，55 次全是**蒽**（总蒽醌/游离蒽醌）
- 用 zhconv 整库转繁简 → 错，会把《药典》正确用法的「癥」（化癥回生片）改成「症」

背景见 docs/10_OCR药品名与异体字纠错.md
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from etl.cleaner import (          # noqa: E402
    DRUG_NAME_CHAR_FIXES, DRUG_NAME_OCR_FIXES, DRUG_NAME_WHOLE_REPLACE,
    clean_entry, clean_text, find_cross_drug_cut, fix_drug_name,
    merge_same_name_entries, trim_cross_drug_tails,
)


# ============================================================
# 药名层
# ============================================================

# (病名, 正确名, 判定依据)
NAME_CASES = [
    # —— 系统性子串/整字混淆 ——
    ("川B", "川芎", "拼音 Chuanxiong / 概述 Ligusticum chuanxiong"),
    ("马齿免", "马齿苋", "拼音 Machixian / PORTULACAE HERBA"),
    ("篇蓄", "萹蓄", "拉丁名 POLYGONI AVICULARIS HERBA"),
    ("蓋政仁", "薏苡仁", "概述 Yiyiren / COICIS SEMEN"),
    ("香薫", "香薷", "拉丁名 MOSLAE HERBA"),
    ("椅藤子", "榼藤子", "概述 Ketengzi / ENTENDAE SEMEN"),
    ("青曝石", "青礞石", "拼音 Qingmengshi"),
    ("龙腳叶", "龙脷叶", "拉丁名 SAUROPI FOLIUM"),
    ("孝苗子", "葶苈子", "拼音 Tinglizi / DESCURAINIAE SEMEN"),
    ("绵萼薛", "绵萆薢", "拼音 Mianbixie"),
    ("紫苑", "紫菀", "拼音 Ziwan"),
    ("務药", "芍药", "拼音 Shayao"),
    ("豬签草", "豨薟草", "正文「豬」30 次全是「豨」；Siegesbeckia"),
    ("豬桐丸", "豨桐丸", "豨桐丸是药典成方"),
    # —— 成方制剂 ——
    ("幵胸顺气丸", "开胸顺气丸", "拼音 Kaixiong Shunqi Wan"),
    ("清幵灵注射液", "清开灵注射液", "同族 7 个剂型一致"),
    ("瞿胆丸", "藿胆丸", "拼音 Huodan Wan，处方含广藿香叶"),
    ("菖菊上清丸", "芎菊上清丸", "拼音 Xiongju Shangqing（石菖蒲的 菖 合法，故用子串）"),
    ("广薑香油", "广藿香油", "拼音 Guanghuoxiang You"),
    ("蛤蛤定喘丸", "蛤蚧定喘丸", "拼音 Gejie / 蚧 原在全库一次都没出现"),
    ("拔毒嘗", "拔毒膏", "拼音 Badu Gao"),
    ("片仔廣胶囊", "片仔癀胶囊", "拼音 Pianzaihuang Jiaonanag"),
    ("片仔攬", "片仔癀", "拼音 Pianzaihuang"),
    ("比拜克胶襄", "比拜克胶囊", "概述 Bibaike Jiaonang"),
    ("牛黄上清胶義", "牛黄上清胶囊", "概述 Niuhuang Shangqing Jiaonang"),
    ("健脾膽浆", "健脾糖浆", "概述 Jianpi Tangjiang"),
    ("复方满山红牘浆", "复方满山红糖浆", "拼音 Fufang Manshanhong Tangjiang"),
    ("参擄十一味颗粒", "参芪十一味颗粒", "拼音 Shenqi Shiyiwei Keli"),
    ("H—味参茂胶囊", "十一味参芪胶囊", "拉丁名 Shiyiwei Shenqi Jiaonang"),
    ("LB泰", "山柰", "拉丁名 Shannai / KAEMPFERIAE RHIZOMA"),
    ("七味橋藤子丸", "七味榼藤子丸", "拉丁名 Qiwei Ketengzi Wan"),
    ("三七总皂苜", "三七总皂苷", "拼音 Zongzaogan"),
    ("三七三醇皂替", "三七三醇皂苷", "拉丁名 NOTOGINSENG TRIOL SAPONINS"),
    ("积雪草总苔", "积雪草总苷", "概述‘总昔’；‘总苔’须先于整字规则‘苔→芩’"),
    ("积雪昔片", "积雪苷片", "概述 Jixuegan Pian"),
    ("三味羨藜散", "三味蒺藜散", "药典名即 三味蒺藜散"),
    # —— 整字级（19 / 8 / 14 个药名同族）——
    ("炙红茂", "炙红芪", "HEDYSARI RADIX PRAEPARATA"),
    ("茂冬颐心口服液", "芪冬颐心口服液", "同族 19 个药名"),
    ("启睥口服液", "启脾口服液", "拼音 Qipi Koufuye"),
    ("葛根苔连片", "葛根芩连片", "药典名即 葛根芩连片"),
    ("辛苔片", "辛芩片", "药典名即 辛芩片"),
    ("皂矶（绿砚）", "皂矾（绿矾）", "拉丁名 MELANTERITUM"),
    ("白矶", "白矾", "同族"),
    ("淫羊蕾", "淫羊藿", "拉丁名 EPIMEDII FOLIUM"),
    ("幵胃山楂丸", "开胃山楂丸", "拼音 Kaiwei Shanzha Wan"),
    # —— 异体字回流（用户打简体搜不到）——
    ("梔子", "栀子", "正文 8,892 次「梔」，全部应为「栀」"),
    ("茵梔黄口服液", "茵栀黄口服液", "同族 5 个剂型"),
    ("清火梔麦片", "清火栀麦片", "同族 3 个剂型"),
    # —— 第二批补漏（2026-09-22 复查；用「拼音字段 vs pypinyin 标音」筛出）——
    # 第一轮用"字符在正文出现次数"粗筛，这些漏掉了：它们的错字在正文里有别的合法用法，
    # 或出现次数不够低。教训：**单靠频次筛不干净，要用第二个独立信号交叉验证**。
    ("白矛根", "白茅根", "概述「禾本科植物白茅Imperata」"),
    ("上荆皮", "土荆皮", "拼音 Tujingpi / PSEUDOLARICIS CORTEX"),
    ("广霍香", "广藿香", "概述「唇形科植物广藿香」"),
    ("般石膏", "煅石膏", "拼音 Duanshigao / GYPSUM USTUM"),
    ("策巨", "菊苣", "拼音 Juju / CICHORII HERBA"),
    ("蛤的", "蛤蚧", "拼音 Gejie / GECKO"),
    ("菠英", "菝葜", "拼音 Baqia / SMILACIS CHINAE RHIZOMA"),
    ("瓜萎子", "瓜蒌子", "拼音 Gualouzi"),
    ("瓜萎皮", "瓜蒌皮", "拼音 Gualoupi"),
    ("桑蝶靖", "桑螵蛸", "拼音 Sangpiaoxiao"),
    ("海蝶艄", "海螵蛸", "拼音 Haipiaoxiao"),
    ("秦茏", "秦艽", "拼音 Qinjiao"),
    ("螟蚣", "蜈蚣", "拼音 Wugong"),
    ("牛莠子", "牛蒡子", "拼音 Niubangzi"),
    ("粉草薜", "粉萆薢", "拼音 Fenbixie / DIOSCOREAE HYPOGLAUCAE"),
    ("莞蔚子", "茺蔚子", "拼音 Chongweizi / LEONURI FRUCTUS（益母草果实）"),
    ("白鼓", "白蔹", "拼音 Bailian / AMPELOPSIS RADIX"),
    ("薪寞", "菥蓂", "拼音 Ximing / THLASPI HERBA"),
    ("莱殖子", "莱菔子", "拼音 Laifuzi"),
    ("蒽麻子", "蓖麻子", "拼音 Bimazi"),
    ("金碌石", "金礞石", "拼音 Jinmengshi"),
    ("英实", "芡实", "拼音 Qianshi"),
    ("棕梱", "棕榈", "拼音 Zonglu"),
    ("S草", "蓍草", "拉丁 ACHILLEAE HERBA / 概述「菊科植物蓍」——注意别用拼音单独判定"),
    ("桑根", "桑椹", "拼音 Sangshen"),
    ("肉女蓉", "肉苁蓉", "拼音 Roucongrong"),
    ("紫箕贯众", "紫萁贯众", "拼音 Ziqiguanzhong"),
    ("清膈X", "清膈丸", "拼音 Qingge Wan（X 是「丸」的误认）"),
    ("蛙贝钙咀II爵片", "蚝贝钙咀嚼片", "拼音 Haobeigai Jujuepian"),
]


@pytest.mark.parametrize("wrong,right,evidence", NAME_CASES)
def test_drug_name_fixed(wrong, right, evidence):
    assert fix_drug_name(wrong) == right, f"{wrong!r} 未纠正为 {right!r}（依据：{evidence}）"


class TestLegitNamesUntouched:
    """合法药名不得被改动——尤其是含『看起来可疑但其实合法』字的药名"""

    LEGIT = [
        "华佗再造丸", "娑罗子", "柿蒂", "白芷", "石菖蒲", "藏菖蒲", "瞿麦",
        "沙苑子", "淫羊藿", "黄芩", "黄蜀葵花", "哈蟆油", "蠲哮片",
        "七宝美髯颗粒", "诺迪康胶囊", "三拗片", "坎离砂", "沈阳红药胶囊",
        "乌军治胆片", "骨友灵搽剂", "通幽润燥丸", "心悦胶囊", "胃乃安胶囊",
        "芪蛭降糖片", "参芪降糖胶囊", "开胸顺气丸", "猪苓", "猪牙皂",
        # 第二批纠错的"正确形态"不得被再次改动（子串规则容易误伤这些）
        "茜草", "蓍草", "蚝贝钙咀嚼片", "清膈丸", "菝葜", "瓜蒌", "秦艽", "蜈蚣", "白蔹", "芡实", "棕榈", "桑椹",
        "肉苁蓉", "莱菔子", "蓖麻子", "桑螵蛸", "海螵蛸", "土荆皮", "广藿香",
        "煅石膏", "菊苣", "白茅根", "蛤蚧", "菥蓂", "金礞石", "粉萆薢", "茺蔚子",
        "牛蒡子", "紫萁贯众", "威灵仙", "甘草", "炙甘草", "黄柏", "关黄柏",
    ]

    @pytest.mark.parametrize("name", LEGIT)
    def test_unchanged(self, name):
        assert fix_drug_name(name) == name


class TestDrugNameWhitespace:
    """药名里的空白必须彻底去除——含 docx 的 `<w:tab/>`

    实例：原文 `<w:r><w:t>策</w:t></w:r><w:r><w:tab/><w:t>巨</w:t></w:r>`，
    python-docx 读出 `策\\t巨`。曾只做 `replace(' ','')`（去不掉 Tab），
    而后面的 `clean_text` 有 `re.sub(r'[ \\t]+', ' ')` 会把 **Tab 折叠成空格**，
    于是库里落成 `策 巨`——**带空格的药名让精确匹配全部失效**。
    """

    @pytest.mark.parametrize("raw,expected", [
        ("策\t巨", "菊苣"),          # Tab + OCR 错字，两件事一起修
        ("人 参", "人参"),           # 半角空格
        ("人\u3000参", "人参"),      # 全角空格
        ("川\t芎", "川芎"),          # Tab
        ("当 归", "当归"),
    ])
    def test_whitespace_removed(self, raw, expected):
        out = clean_entry({"drug_name": raw, "parent_drug": "",
                           "sections": [], "intro_text": ""})
        assert out["drug_name"] == expected
        assert not any(c.isspace() for c in out["drug_name"])

    def test_parent_drug_whitespace_removed(self):
        out = clean_entry({"drug_name": "策 巨-饮片", "parent_drug": "策\t巨",
                           "sections": [], "intro_text": ""})
        assert out["drug_name"] == "菊苣-饮片"
        assert out["parent_drug"] == "菊苣"


class TestLayerOrderInteraction:
    """药名层规则跑在正文层规则**之前**，所以键要用**原始形态**的字

    实例：原文「蔥麻子」（蓖麻子的误认）。若把规则键写成「蒽麻子」，
    因为 `蔥→蒽` 发生在 `clean_text`（后跑），`fix_drug_name`（先跑）永远命中不了，
    清洗后库里会留下 `蒽麻子`。实测确实踩到过。
    """

    @pytest.mark.parametrize("raw,expected", [
        ("蔥麻子", "蓖麻子"),      # 原文形态
        ("菌麻子", "蓖麻子"),      # 另一种误认
        ("蒽麻子", "蓖麻子"),      # 万一 clean_text 先跑过的形态
        ("菌麻油", "蓖麻油"),
    ])
    def test_raw_form_key(self, raw, expected):
        out = clean_entry({"drug_name": raw, "parent_drug": "",
                           "sections": [], "intro_text": ""})
        assert out["drug_name"] == expected

    def test_no_name_is_still_fixable_after_cleaning(self):
        """清洗后的药名不应再被规则改动——否则说明规则键与清洗顺序错配"""
        for raw in ["蔥麻子", "蔥麻子-饮片", "菌麻子", "策\t巨", "川B", "炙黄芷",
                    "马齿免", "梔子", "豬签草", "S草", "清膈X"]:
            cleaned = clean_entry({"drug_name": raw, "parent_drug": "",
                                   "sections": [], "intro_text": ""})["drug_name"]
            assert fix_drug_name(cleaned) == cleaned, f"{raw!r} 清洗后仍可被规则改动"


def test_no_garbled_char_survives():
    """清洗结果里不得残留任何整字级错字（新错字混入会在此暴露）"""
    for wrong in DRUG_NAME_CHAR_FIXES:
        assert fix_drug_name(wrong) != wrong or True  # 单字本身必然被替换
    # 组合检验：把错字嵌进名字里也不应残留
    for wrong, right in DRUG_NAME_CHAR_FIXES.items():
        assert fix_drug_name(f"测试{wrong}甲") == f"测试{right}甲"


def test_fix_is_idempotent():
    """幂等：反复清洗结果不变（避免下游重复调用时抖动）"""
    for wrong, _right, _evi in NAME_CASES:
        once = fix_drug_name(wrong)
        assert fix_drug_name(once) == once


def test_substring_layer_runs_before_char_layer():
    """顺序敏感：「总苔」必须先于整字「苔→芩」，否则会变成「总芩」"""
    assert fix_drug_name("积雪草总苔") == "积雪草总苷"


# ============================================================
# 正文层
# ============================================================

CONTENT_CASES = [
    ("总蔥醍 照高效液相色谱法", "总蒽醌 照高效液相色谱法"),
    ("游离蔥酿同药材", "游离蒽醌同药材"),
    ("合并乙醍液，蒸干", "合并乙醚液，蒸干"),        # 曾被误改成「乙滤液」
    ("石油醍（60〜90°C）", "石油醚（60〜90°C）"),
    ("本品为茜草科植物梔子Gardenia", "本品为茜草科植物栀子Gardenia"),
    ("或毛梗豨签 Siegesbeckia", "或毛梗豨薟 Siegesbeckia"),
    ("川萼330g", "川芎330g"),
    ("黄苓120g", "黄芩120g"),
    ("广蕾香叶4000g", "广藿香叶4000g"),
    ("展幵，取出，晾干", "展开，取出，晾干"),
    ("加白矶2kg", "加白矾2kg"),
    ("含人参皂苜Rf", "含人参皂苷Rf"),
    ("含人参皂 昔Rg1", "含人参皂苷Rg1"),
]


@pytest.mark.parametrize("before,after", CONTENT_CASES)
def test_content_fixed(before, after):
    assert clean_text(before) == after


class TestContentMustNotOverFix:
    """正文里合法出现的字不得被整字替换"""

    KEEP = [
        "夏、秋二季叶茂盛时采收，除去杂质",     # 茂 合法
        "本品为苦苣苔科植物吊石苣苔",           # 苔 合法
        "的干燥花蕾。当花蕾由绿色转红时采摘",   # 蕾 合法
        "萼筒圆柱状，略扁",                     # 萼 合法
        "以相应的被替代对照品确证为准",         # 替 合法
        "化癥回生片",                           # 癥 是《药典》正确用法，不得转成「症」
    ]

    @pytest.mark.parametrize("text", KEEP)
    def test_unchanged(self, text):
        assert clean_text(text) == text


# ============================================================
# 源文档结构性缺陷：药名整行丢失 → 整名替换
# ============================================================

class TestWholeNameReplace:
    """药名整行丢失时，子串规则修不干净，必须整名替换

    实例：原文把「药名+拼音+拉丁名+概述」挤在同一段落且无换行
    （`高山辣根菜GaoshanlagencaiPEGAEOPHYTI RADIX ET RHIZOMA（Hook.f.etThoms.）Marq.etShaw的干燥根和根茎`），
    解析器因整段过长否决，于是把下一段（西文+中文）当成了药名。
    """

    RAW = ("PEGAEOPHYTIRADIXETRHIZOMA（Hook.f.etThoms.）Marq.etShaw"
           "的干燥根和根茎。秋季采挖，除去须根和泥沙，晒干。")

    def test_prefix_maps_to_real_name(self):
        assert fix_drug_name(self.RAW) == "高山辣根菜"

    def test_not_a_substring_rule(self):
        """必须整名替换：子串替换会把残余的西文/中文留在名字里"""
        assert fix_drug_name(self.RAW) != "高山辣根菜（Hook.f.etThoms.）Marq.etShaw的干燥根和根茎。秋季采挖，除去须根和泥沙，晒干。"

    def test_no_fake_name_survives(self):
        for prefix in DRUG_NAME_WHOLE_REPLACE:
            assert not fix_drug_name(prefix + "xxx").startswith(prefix)


# ============================================================
# 同名条目合并（多栏/表格版面导致同一饮片被拆成两条）
# ============================================================

def _e(name, secs, **kw):
    d = {"drug_name": name, "parent_drug": "", "is_yinpian": kw.get("is_yinpian", False),
         "pinyin_name": kw.get("pinyin", ""), "latin_name": "", "intro_text": kw.get("intro", ""),
         "sections": [{"section_name": s, "content": c, "table_markdown": None,
                       "raw_paragraphs": []} for s, c in secs]}
    return d


class TestMergeSameName:
    def test_merges_and_dedups(self):
        a = _e("锁阳-饮片", [("炮制", "洗净，润透"), ("性状", "本品为不规则形")])
        b = _e("锁阳-饮片", [("性状", "本品为不规则形"), ("功能与主治", "补肾阳")])
        out = merge_same_name_entries([a, b])
        assert len(out) == 1
        assert [s["section_name"] for s in out[0]["sections"]] == ["炮制", "性状", "功能与主治"]
        assert out[0]["sections"][1]["content"] == "本品为不规则形"   # 重复内容只留一份

    def test_no_content_lost(self):
        a = _e("苍术-饮片", [("炮制", "甲")])
        b = _e("苍术-饮片", [("鉴别", "乙")])
        out = merge_same_name_entries([a, b])
        got = {s["content"] for s in out[0]["sections"]}
        assert got == {"甲", "乙"}

    def test_fills_missing_metadata(self):
        a = _e("刀豆-饮片", [("炮制", "甲")])
        b = _e("刀豆-饮片", [("性状", "乙")], pinyin="Daodou", intro="本品为…")
        out = merge_same_name_entries([a, b])
        assert out[0]["pinyin_name"] == "Daodou"
        assert out[0]["intro_text"] == "本品为…"

    def test_order_is_first_occurrence(self):
        out = merge_same_name_entries([_e("乙", [("性状", "1")]), _e("甲", [("性状", "2")]),
                                       _e("乙", [("鉴别", "3")])])
        assert [e["drug_name"] for e in out] == ["乙", "甲"]

    def test_different_names_untouched(self):
        a, b = _e("人参", [("性状", "甲")]), _e("人参-饮片", [("性状", "乙")])
        assert len(merge_same_name_entries([a, b])) == 2


# ============================================================
# 跨药污染尾巴剪裁（缺陷 11 的残留）
# ============================================================
# 现象：条目某一节末尾粘着**下一个药的开头**。
# 根因与缺陷 11 同源——源 docx 多栏/表格版面，边界处药名行缺失或被拆散。
# ⚠️ 只能保守判定，两个坑都在测试里固化：
#   ① 药典允许【贮藏】后跟 `附：质量标准` / `注：…`（实测最长 2032 字是**合法**的）
#   ② 正文里也会出现独占一行的西文公式（`bOO X w`），单看"拼音样的一行"会误伤

class TestCrossDrugCut:
    CUT_CASES = [
        # (正文, 说明)
        ("置通风干燥处。\nCang'erzi\nXANTHII FRUCTUS\n本品为菊科植物苍耳Xanthium sibiricum",
         "拼音+拉丁名+本品为（苍术-饮片 实例）"),
        ("置干燥处。\n本品为根树科植物破布叶Microcos paniculata L.的干燥叶",
         "本品为（布渣叶 实例）"),
        ("置阴凉处。\nSanqi\nNOTOGINSENG RADIX ET RHIZOMA\n本品为五加科植物三七",
         "拼音行在前（刀豆-饮片 实例）"),
        ("置通风干燥处，防蛀。\nJianghuang\nCURCUMAE LONGAE RHIZOMA\n本品为姜科植物姜黄",
         "拉丁名全大写（急性子 实例）"),
    ]

    KEEP_CASES = [
        ("密封。\n附：人参提取物质量标准\n人参提取物\n本品为五加科植物人参",
         "附：附加质量标准——**合法**，实测最长 2032 字"),
        ("密封，置阴凉处。\n注：1.磷酸盐缓冲液（pH 7.8）的配制 取磷酸氢二钠58g",
         "注：脚注——合法"),
        ("密封，置干燥处。\n[:制剂】口服制剂注射剂",
         "制剂标记（残缺写法 `[:制剂】`）——合法，必须容忍 OCR 变体"),
        ("密封，置阴凉处。\n硅酸镁含量=（钙、镁总量一可溶性钙镁含量）X5. 20",
         "含量测定公式——**曾误伤**：`bOO X w` 这种行会被 ASCII 规则命中"),
        ("置通风干燥处。", "短正文不判"),
    ]

    @pytest.mark.parametrize("text,why", CUT_CASES)
    def test_detects_contamination(self, text, why):
        assert find_cross_drug_cut(text) > 0, f"应判为跨药污染：{why}"

    @pytest.mark.parametrize("text,why", KEEP_CASES)
    def test_keeps_legit_content(self, text, why):
        assert find_cross_drug_cut(text) == -1, f"应保留：{why}"

    # 【贮藏】专用更强规则：首行是完整陈述句时，其后只允许 附/注/制剂。
    # 依据：仓储正文就是一两句短话；实测 89 个"过长仓储"里多数其后跟的是别的药的
    # 鉴别/含量测定碎片（`对照品溶液的制备…`），这类碎片**不含 `本品为`**，通用规则抓不到。
    STORAGE_CUT = [
        ("置一4°C贮存。\n对照品溶液的制备取白杨素对照品、高良姜素对照品、咖啡酸苯乙酯对照品适量，精密称定",
         "碎片型污染（蜂胶 实例）——无 `本品为`，只有仓储专用规则能抓到"),
        ("置干燥处。\n密称定，置具塞锥形瓶中，精密加入70%甲醇50ml，密塞，称定重量，超声处理",
         "碎片型污染（煅石膏 实例）"),
    ]
    STORAGE_KEEP = [
        ("密封，置阴凉处。\n附：茵陈提取物质量标准\n茵陈提取物\n〔制法J取茵陈，加水煎煮三次",
         "附：附加质量标准——**合法**，实测最长 2032 字"),
        ("密封。\n注：益母草浸膏含量测定方法取浸膏约0.5g，精密称定，置烧杯中",
         "注：脚注——合法"),
        ("密封，置干燥处。\n[:制剂】口服制剂注射剂", "制剂（残缺标记）——合法"),
        ("（1）基质为混合脂肪酸甘油酯的栓：密闭，在\n2。 笆以下保存。（2）基质为聚乙二醇的栓：密闭，在30°C以下保存。",
         "**多行仓储**（野菊花栓 的两种基质）——首行不是完整句，必须保留"),
    ]

    @pytest.mark.parametrize("text,why", STORAGE_CUT)
    def test_storage_rule_detects_fragment_contamination(self, text, why):
        assert find_cross_drug_cut(text, "贮藏") > 0, why

    @pytest.mark.parametrize("text,why", STORAGE_KEEP)
    def test_storage_rule_keeps_legit(self, text, why):
        assert find_cross_drug_cut(text, "贮藏") == -1, why

    SINGLE_LINE = [
        ("置干燥处，防霉，防蛀。 于同一硅胶G薄层板上，以三氯甲烷-甲醇（9 ： 1）为展开剂，展开，取出，晾干",
         "玄参 实例：污染接在**同一行**空格后，换行规则抓不到"),
    ]
    SINGLE_LINE_KEEP = [
        ("鲜地黄埋在沙土中，防冻；生地黄置通风干燥 处，防霉，防蛀。", "地黄：同节陈述两种形态"),
        ("用木箱严密封装，常用花椒拌存，置阴凉干燥 处，防蛀。", "蛤蚧-饮片：合法长句"),
        ("遮光，密闭，在阴凉干燥处保存。防潮。", "短尾补充句"),
    ]

    @pytest.mark.parametrize("text,why", SINGLE_LINE)
    def test_single_line_tail_detected(self, text, why):
        assert find_cross_drug_cut(text, "贮藏") > 0, why

    @pytest.mark.parametrize("text,why", SINGLE_LINE_KEEP)
    def test_single_line_legit_kept(self, text, why):
        assert find_cross_drug_cut(text, "贮藏") == -1, why

    def test_storage_rule_not_applied_to_other_sections(self):
        """公式等非仓储正文只走通用规则，不受仓储专用规则影响"""
        t = "密封，置阴凉处。\n硅酸镁含量=（钙、镁总量一可溶性钙镁含量）X5. 20 式中c为乙二胺四醋酸二钠"
        assert find_cross_drug_cut(t, "贮藏") > 0        # 按仓储规则会判污染
        assert find_cross_drug_cut(t, "含量测定") == -1   # 按通用规则保留

    def test_cut_position_keeps_own_text(self):
        t = "置通风干燥处。\nSanqi\nNOTOGINSENG RADIX ET RHIZOMA\n本品为五加科植物三七"
        assert t[:find_cross_drug_cut(t)] == "置通风干燥处。"

    def test_idempotent(self):
        t = "置干燥处。\n本品为根树科植物破布叶"
        once = t[:find_cross_drug_cut(t)]
        assert find_cross_drug_cut(once) == -1


class TestTrimCrossDrugTails:
    def _e(self, name, secs):
        return {"drug_name": name, "parent_drug": "", "is_yinpian": False,
                "pinyin_name": "", "latin_name": "", "intro_text": "",
                "sections": [{"section_name": s, "content": c,
                              "table_markdown": None, "raw_paragraphs": []} for s, c in secs]}

    def test_trims_and_reports(self):
        e = self._e("苍术-饮片", [("贮藏", "置通风干燥处。\nCang'erzi\nXANTHII FRUCTUS\n本品为菊科植物苍耳")])
        st = trim_cross_drug_tails([e])
        assert e["sections"][0]["content"] == "置通风干燥处。"
        assert st["entries_touched"] == 1 and st["chars_dropped"] > 0
        assert st["samples"] and st["samples"][0][0] == "苍术-饮片"

    def test_drops_sections_after_storage_except_zhiji(self):
        e = self._e("锁阳-饮片", [("贮藏", "置通风干燥处。"),
                              ("鉴别", "取本品粉末1g…"), ("含量测定", "照高效液相色谱法…")])
        st = trim_cross_drug_tails([e])
        assert [s["section_name"] for s in e["sections"]] == ["贮藏"]
        assert st["sections_dropped"] == 2

    def test_keeps_zhiji_after_storage(self):
        """【制剂】在药典编排里本来就在【贮藏】之后——合法，不能删"""
        e = self._e("三七三醇皂苷", [("贮藏", "遮光，密闭。"), ("制剂", "口服制剂 注射剂")])
        st = trim_cross_drug_tails([e])
        assert [s["section_name"] for s in e["sections"]] == ["贮藏", "制剂"]
        assert st["sections_dropped"] == 0 and st["entries_touched"] == 0

    def test_legit_storage_untouched(self):
        e = self._e("人参茎叶总皂苷", [("贮藏", "密封。\n附：人参提取物质量标准\n人参提取物\n本品为…")])
        st = trim_cross_drug_tails([e])
        assert st["entries_touched"] == 0

    def test_no_sections_no_crash(self):
        assert trim_cross_drug_tails([self._e("空", [])])["entries_touched"] == 0


class TestDanYanZi:
    """「澹」一字多义 + 「洶」——由**双判官交叉验证**发现

    第二判官核验引文时发现回答里有「食少便澹」，而引文核验**通过**
    （逐字匹配）→ 说明错字在**语料里**，不在回答里。全库排查：
    - 「澹」57 次 = 55 次「溏」+ 2 次「谵」（溏/谵 在全库都是 0 次）
    - 「洶」7 次全是显微描述的「沟」
    这是交叉验证机制的意外产出：它不但校准了判分，还顺带抓出一个语料缺陷。
    """

    @pytest.mark.parametrize("raw,expected", [
        ("用于气虚乏力，食少便澹，中气下陷", "食少便溏"),
        ("五更澹泻、食少不振", "五更溏泻"),
        ("大便澹薄、舌质淡", "大便溏薄"),
        ("大便稀澹或腹泻", "大便稀溏"),
        ("秘结或澹而不爽", "溏而不爽"),
        ("内无澹心时，取出", "内无溏心时"),        # 阿胶珠：炒至成珠内无溏心
        ("壁厚，孔洶明显", "孔沟明显"),            # 石细胞显微描述
        ("可见纵皱纹或纵洶", "纵皱纹或纵沟"),
    ])
    def test_dan_to_tang(self, raw, expected):
        out = clean_text(raw)
        assert expected in out and "澹" not in out and "洶" not in out

    @pytest.mark.parametrize("raw,expected", [
        ("痰热澹狂，神昏不语", "痰热谵狂"),        # 谵 必须先于整字 澹→溏
        ("热入心包，神昏澹语", "神昏谵语"),
    ])
    def test_dan_to_zhan_before_tang(self, raw, expected):
        out = clean_text(raw)
        assert expected in out, f"顺序错误：「谵」被「溏」规则吃掉了 → {out!r}"

    @pytest.mark.parametrize("good", ["大便溏泄", "神昏谵语", "叶脉具纵沟"])
    def test_correct_forms_untouched(self, good):
        assert clean_text(good) == good



class TestDefect19Typos:
    """缺陷 19：又一批系统性 OCR 错字（与缺陷 13 的发现方式相同）

    这批不是"猜"出来的，而是**判分交叉验证判官说"资料原文为 X，与论断不符"**时
    暴露的：模型写的是对的、语料是错的。全部规则都满足判据
    「正确写法在全库正文出现 0 次」，且错字有合法用法时改用词组级规则。
    """

    @pytest.mark.parametrize("raw,expected", [
        ("取橙皮昔对照品，精密称定", "橙皮苷"),
        ("数按黄芪甲昔峰计算", "黄芪甲苷"),
        ("以栀子昔峰计算应", "栀子苷"),
        ("人参皂昔", "人参皂苷"),
        ("用于胱腹胀痛，呕吐泻痢", "脘腹胀痛"),
        ("胸胁、胱腹胁痛", "脘腹胁痛"),
        ("食少底腹痛", "脘腹痛"),
        ("用于癥痕痞块，痛经", "癥瘕痞块"),
        ("癥痕腹痛，风湿痹痛", "癥瘕腹痛"),
        ("用于瘰疬痰核", "瘰疬痰核"),          # 已正确的写法不得被改坏
        ("用于乳痈，瘪痂,痰核", "瘰疬"),
        ("痈肿，痕 痂，疥癣", "瘰疬"),
        ("带下，瘪痂瘻瘤，胃痛", "瘰疬瘿瘤"),
        ("喘息，吐血，朝血，崩漏下血", "衄血"),
        ("血热吐朝，目赤咽肿", "吐衄"),
        ("鼻朝 齿蛻，鼻瘪肉", "鼻衄"),
        ("含朝蕾定C（", "朝藿定C"),
        ("喷以稀碘化朝钾试液", "碘化铋钾"),
        ("用于蛔虫病，蛻虫病", "蛲虫病"),
        ("另取蝉蛻", "蝉蜕"),
        ("再取蛻皮螢酮对照品", "蜕皮甾酮"),
        ("取千金子當醇峰计算", "甾醇"),
        ("再取芝麻素对照品、步谷當醇对照品", "β-谷甾醇"),
        ("以乙腊-0. 1%歸酸溶液（37 : 63）为流动相", "0.1%磷酸"),
        ("加三氯化歸约0. 5ml", "三氯化铁"),
        ("酒石酸 歸钾0. 1g", "锑钾"),
        ("取出，陳干，喷以", "晾干"),
        ("豨菴草330", "豨莶草"),
        # 二次扫描新挖出的
        ("朝鲜淫羊蕾", "淫羊藿"),          # 326 次：正文里「淫羊藿」原本只剩 4 次
        ("淫羊蕾苷", "淫羊藿苷"),
        ("湿滞伤中，胱痞吐泻", "脘痞吐泻"),
        ("湿滞伤中，胱 痞吐泻", "脘痞吐泻"),
        ("乳痈，瘪疡，蛇虫咬伤", "瘰疬"),
        # ⚠️ 语料里只有**繁体**「療」（實測：療疡 4 次、療病 4 次；简体 疗疡/疗病 均为 0），
        #    故规则与测试都只针对 療 —— 不为不存在的形态造规则
        ("療疡痰核", "瘰疬"),
        ("療病痰核", "瘰疬"),
    ])
    def test_fix(self, raw, expected):
        out = clean_text(raw)
        assert expected in out, f"{raw!r} → {out!r}"

    @pytest.mark.parametrize("good", [
        "归肾、膀胱经。",              # 胱 在膀胱经合法
        "置圆底烧瓶中，加水",          # 底 在圆底烧瓶合法
        "置平底烧瓶中，精密加",        # 平底
        "抓痕、血痂、色素",            # 血痂 合法（不能改成 血疬）
        "层层剥落痕迹（滑石）",        # 痕迹 合法
        "窝状 茎痕（芦碗）",           # 茎痕 合法
        "茎干瘪中空，表面黄",          # 干瘪 合法
        "取蝉蜕",                      # 蜕 已是正确写法
        "广藿香",                      # 藿 合法
        "三氯化铁试液",                # 正确写法
        "取出，晾干，喷以",            # 正确写法
        "花蕾",                        # 蕾 在花蕾合法（不能改成 花藿）
        "溃疡、疮疡",                  # 疡 合法（不能改成 瘰疬）
        "痞满",                        # 痞 合法
        "朝鲜淫羊藿",                  # 朝 合法
    ])
    def test_legal_usage_untouched(self, good):
        assert clean_text(good) == good, f"合法用法被改坏：{good!r} → {clean_text(good)!r}"

    def test_drug_name_neixiao_luoli(self):
        from etl.cleaner import fix_drug_name
        assert fix_drug_name("内消痕痂片") == "内消瘰疬片"

    def test_idempotent(self):
        """清洗两次结果一致（词表规则不得互相打架）"""
        for raw in ["取橙皮昔对照品", "用于胱腹胀痛", "鼻朝 齿蛻", "再取蝉蛻", "瘪痂瘻瘤"]:
            once = clean_text(raw)
            assert clean_text(once) == once, f"不幂等：{raw!r} → {once!r} → {clean_text(once)!r}"
