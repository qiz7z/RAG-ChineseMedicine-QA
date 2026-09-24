# -*- coding: utf-8 -*-
"""
数据清洗器
================
对解析后的药品数据进行清洗：
1. OCR 错别字纠错（【検查】→【检查】等）
2. 全角/半角统一
3. 多余空白合并
4. 异常字符清理
"""
import re
import json
from pathlib import Path


# ============================================================
# OCR 错别字映射表
# ============================================================

# 章节标记级错误（从数据分析中发现）
SECTION_OCR_FIXES = {
    '【検查】': '【检查】',
    '【含量測定】': '【含量测定】',
    '〔礎藏〕': '〔贮藏〕',
    '〔疋藏〕': '〔贮藏〕',
    '〔规格（l）〕': '〔规格（1）〕',
    '【性味与帰経】': '【性味与归经】',
    '【性味與歸經】': '【性味与归经】',
}

# 章节名称纠错（解析后section_name已去除【】括号，需单独处理）
SECTION_NAME_NORMALIZE = {
    # 检查
    '検查': '检查', '检査': '检查',
    # 含量测定
    '含量測定': '含量测定', '含量则定': '含量测定',
    # 性味与归经
    '性昧与归经': '性味与归经', '性味與歸經': '性味与归经',
    # 贮藏（各种OCR变体）
    'St藏': '贮藏', '/藏': '贮藏', '巫藏': '贮藏', '旋藏': '贮藏',
    '疋藏': '贮藏', '礎藏': '贮藏', '触藏': '贮藏', '正藏': '贮藏',
    # 浸出物
    '浸岀物': '浸出物',
    # 功能主治
    '功能主治': '功能与主治',
    # 用法
    '用法': '用法与用量',
    # 注意
    '注意事项': '注意',
}

# ============================================================
# 药品名 OCR 纠正（两层，**先子串、后整字**，顺序敏感）
# ============================================================
# 背景：原始 DOCX 的文字层 OCR 把一批近形字系统性认错，规律例如
#   芪→茂   芩→苔/苓   芎→萼/菖/弯/B   脾→睥   开→幵
#   藿→蕾/薑  矾→矶/砚   苷→苜/替/昔   糖→牘/膽   囊→襄/義
# 这些错字**只在药品名里**不合法，所以整字替换只在 drug_name / parent_drug 上做。
# （正文另见 CONTEXT_FIXES：茂/苔/蕾/萼/替 在正文里合法出现，例如「叶茂盛时采收」
#  「苦苣苔科」「干燥花蕾」「萼筒」「被替代对照品」，故正文**不能**做整字替换。）

# ---- 第 1 层：子串级（仅在该组合出现时替换，安全，优先执行）----
# 之所以必须先于整字规则：'总苔'→'总苷' 要让位于整字规则 '苔'→'芩'，
# 否则「积雪草总苔」会被错成「积雪草总芩」。
DRUG_NAME_OCR_FIXES = {
    # 单味制剂 / 提取物
    'H—味参茂': '十一味参芪',    # 拉丁名 Shiyiwei Shenqi Jiaonang
    'LB泰': '山柰',              # 拉丁名 Shannai / KAEMPFERIAE RHIZOMA
    '绵萼薛': '绵萆薢',          # 拼音 Mianbixie
    '紫苑': '紫菀',              # 拼音 Ziwan（'沙苑子' 的 苑 合法，故用子串）
    '积雪昔片': '积雪苷片',       # 概述 Jixuegan Pian
    '积雪草总苔': '积雪草总苷',
    '皂苜': '皂苷',
    '皂替': '皂苷',
    # 成方制剂
    '片仔廣': '片仔癀',
    '片仔攬': '片仔癀',
    '胶襄': '胶囊',
    '胶義': '胶囊',
    '牘浆': '糖浆',
    '膽浆': '糖浆',
    '拔毒嘗': '拔毒膏',
    '定鬧丸': '定喘丸',
    '广薑香油': '广藿香油',
    '瞿胆丸': '藿胆丸',          # 拼音 Huodan Wan，处方含广藿香叶
    '菖菊上清': '芎菊上清',       # 拼音 Xiongju Shangqing（'石菖蒲/藏菖蒲' 的 菖 合法）
    '参擄十一味': '参芪十一味',
    '茂苗强心': '芪苈强心',
    # 药材 / 饮片
    '椅藤子': '榼藤子',          # 概述 Ketengzi / ENTADAE SEMEN
    '橋藤子': '榼藤子',
    '蓋政仁': '薏苡仁',          # 概述 Yiyiren / COICIS SEMEN
    '篇蓄': '萹蓄',              # 拉丁名 POLYGONI AVICULARIS HERBA
    '马齿免': '马齿苋',          # 拼音 Machixian / PORTULACAE HERBA
    '香薫': '香薷',              # 拉丁名 MOSLAE HERBA
    '龙腳叶': '龙脷叶',          # 拉丁名 SAUROPI FOLIUM
    '青曝石': '青礞石',          # 拼音 Qingmengshi
    '孝苗子': '葶苈子',          # 拼音 Tinglizi / DESCURAINIAE SEMEN
    '務药': '芍药',              # 拼音 Shayao
    '川B': '川芎',               # 拼音 Chuanxiong / Ligusticum chuanxiong
    '参芷降糖': '参芪降糖',
    '芷蛭降糖': '芪蛭降糖',
    '炙黄芷': '炙黄芪',
    '黄芷': '黄芪',
    '川萼': '川芎',
    '主茯苓': '土茯苓',
    '蛤蛤': '蛤蚧',              # 蛤蛤定喘丸 / 蛤蛤补肾胶囊（蚧 在全库药名中原本一次都没出现）
    '羨藜': '蒺藜',              # 三味羨藜散 → 三味蒺藜散
    '豬签草': '豨薟草',           # 豨薟草（Siegesbeckia），正文里「豬」30 次全是「豨」
    '豬签丸': '豨薟丸',
    '豬莅通栓': '豨薟通栓',
    '稀签通栓': '豨薟通栓',       # 同一制剂名的另一种误认写法
    '豬桐': '豨桐',              # 豬桐丸 / 豬桐胶囊 → 豨桐丸 / 豨桐胶囊
    # ---- 第二批（2026-09-22 复查补漏：用「拼音字段 vs pypinyin 标音」筛出）----
    # 这些是第一轮漏掉的：第一轮靠"字符在正文中出现次数"粗筛，而它们的错字在正文里
    # 有别的合法用法或出现次数不够低，所以没被筛出来。
    '白矛根': '白茅根',           # 概述「禾本科植物白茅Imperata」
    '上荆皮': '土荆皮',           # 拼音 Tujingpi / PSEUDOLARICIS CORTEX
    '广霍香': '广藿香',           # 概述「唇形科植物广藿香」
    '般石膏': '煅石膏',           # 拼音 Duanshigao / GYPSUM USTUM
    '策巨': '菊苣',               # 拼音 Juju / CICHORII HERBA（原名含 <w:tab/>）
    '蛤的': '蛤蚧',               # 拼音 Gejie / GECKO
    '菠英': '菝葜',               # 拼音 Baqia / SMILACIS CHINAE RHIZOMA
    '瓜萎': '瓜蒌',               # 拼音 Gualou（覆盖 瓜萎子 / 瓜萎皮）
    '桑蝶靖': '桑螵蛸',           # 拼音 Sangpiaoxiao
    '海蝶艄': '海螵蛸',           # 拼音 Haipiaoxiao
    '秦茏': '秦艽',               # 拼音 Qinjiao
    '螟蚣': '蜈蚣',               # 拼音 Wugong
    '牛莠子': '牛蒡子',           # 拼音 Niubangzi
    '粉草薜': '粉萆薢',           # 拼音 Fenbixie / DIOSCOREAE HYPOGLAUCAE
    '莞蔚子': '茺蔚子',           # 拼音 Chongweizi / LEONURI FRUCTUS（益母草果实）
    '白鼓': '白蔹',               # 拼音 Bailian / AMPELOPSIS RADIX
    '薪寞': '菥蓂',               # 拼音 Ximing / THLASPI HERBA
    '莱殖子': '莱菔子',           # 拼音 Laifuzi
    # ⚠️ 键必须用**原始形态**的字：fix_drug_name 跑在 clean_text **之前**，
    #    而正文层的 `蔥→蒽` 在 clean_text 里。所以原文「蔥麻子」若把键写成
    #    「蒽麻子」就永远命中不了——实测确实踩到：清洗后库里留下 `蒽麻子`。
    #    与「菌麻油→蓖麻油」同源（蓖 被认成 蔥 / 菌）。
    '蔥麻子': '蓖麻子',
    '菌麻子': '蓖麻子',
    '蒽麻子': '蓖麻子',           # 拼音 Bimazi
    '金碌石': '金礞石',           # 拼音 Jinmengshi（与「青曝石→青礞石」同源）
    '英实': '芡实',               # 拼音 Qianshi
    '棕梱': '棕榈',               # 拼音 Zonglu
    'S草': '蓍草',               # 拼音 Shicao / **ACHILLEAE HERBA**，概述「菊科植物蓍」
                                  # ⚠️ 注意：这里**不能**用拼音单独判定。同音的「茜草」
                                  #   是另一个药（拉丁 RUBIAE RADIX）。曾误标成「茜草」，
                                  #   结果库里出现两个「茜草」，靠拉丁名才发现。
    '桑根': '桑椹',               # 拼音 Sangshen
    '肉女蓉': '肉苁蓉',           # 拼音 Roucongrong
    '紫箕贯众': '紫萁贯众',       # 拼音 Ziqiguanzhong
    '清膈X': '清膈丸',            # 拼音 Qingge Wan（X 是「丸」的误认）
    '蛙贝钙咀II爵片': '蚝贝钙咀嚼片',  # 拼音 Haobeigai Jujuepian；「嚼」被拆成 II+爵
}

# ---- 第 0 层：整名替换（源文档结构性缺陷，非 OCR 错字）----
# 这些条目的药名**整行丢失**，只剩下一段西文+中文描述被当成了药名。
# 子串规则无法修复（替换掉匹配部分后，残余文字还会留在名字里），必须整名替换。
# 判定依据：拉丁名与原文中残留的中文名可互证。
DRUG_NAME_WHOLE_REPLACE = {
    # 高山辣根菜：原文药名行把「药名+拼音+拉丁名+概述」挤在同一段落且无换行
    #   `高山辣根菜GaoshanlagencaiPEGAEOPHYTI RADIX ET RHIZOMA（Hook.f.etThoms.）Marq.etShaw的干燥根和根茎`
    # → `_is_likely_drug_name` 因整段过长而否决，于是下一段（西文+中文）被当成药名。
    # 修后该条目恢复为「高山辣根菜」，拉丁名 PEGAEOPHYTI RADIX ET RHIZOMA 可互证。
    'PEGAEOPHYTIRADIXETRHIZOMA': '高山辣根菜',
}

# ---- 第 2 层：整字级（该错字在全部 2,428 个药名中从不合法出现）----
# 判定方法：全库扫描该字只出现在病名里，且拼音名 / 拉丁名 / 正文三者互证。
DRUG_NAME_CHAR_FIXES = {
    '茂': '芪',    # 参茂→参芪、炙红茂→炙红芪、茂冬颐心→芪冬颐心…（19 个药名）
    '睥': '脾',    # 健睥→健脾、启睥→启脾…（8 个药名）
    '幵': '开',    # 幵胸顺气→开胸顺气、清幵灵→清开灵…（14 个药名）
    '蕾': '藿',    # 淫羊蕾→淫羊藿、蕾香正气→藿香正气…（不覆盖 石菖蒲 的 菖）
    '苔': '芩',    # 葛根苔连→葛根芩连、辛苔片→辛芩片…（子串层已先处理 总苔）
    '矶': '矾',    # 皂矶→皂矾、白矶→白矾
    '砚': '矾',    # 绿砚→绿矾
    '昔': '苷',    # 积雪昔→积雪苷
    '苜': '苷',    # 皂苜→皂苷
    '梔': '栀',    # 异体字回流：梔子→栀子、茵梔黄→茵栀黄、清火梔麦→清火栀麦（13 个药名）
    # 缺陷 19：内消痕痂片 → **内消瘰疬片**（真实中成药名）。
    # 全库 2,421 个药名中，「痂」「痕」只出现在这一个药名里 → 可用整字级。
    '痂': '疬',
    '痕': '瘰',
}

# 字符级错误（常见 OCR 误识别）
CHAR_OCR_FIXES = {
    '昔': '苷',     # 人参皂昔 → 人参皂苷（非常常见的OCR错误）
    '甘': '苷',     # 部分上下文中 甘 应为 苷（但需注意"甘"本身也是药味描述，只在特定上下文替换）
    '蔥': '蒽',     # 蒽醌类
    '月青': '肼',   # 
    '检査': '检查',
    '検查': '检查',
    '測定': '测定',
}

def fix_drug_name(name: str) -> str:
    """对药品名应用三层纠正：整名 → 子串级 → 整字级（顺序不可颠倒）。

    - 第 0 层「整名替换」必须最先做，且用 startswith 判定：源文档结构性缺陷下
      药名整行丢失，子串替换会把残余的西文/中文留在名字里。
    """
    for prefix, correct in DRUG_NAME_WHOLE_REPLACE.items():
        if name.startswith(prefix):
            return correct
    for wrong, correct in DRUG_NAME_OCR_FIXES.items():
        if wrong in name:
            name = name.replace(wrong, correct)
    for wrong, correct in DRUG_NAME_CHAR_FIXES.items():
        if wrong in name:
            name = name.replace(wrong, correct)
    return name


# 正文级替换规则
# (正则模式, 替换文本)
# ⚠️ 正文与药名**不能**共用整字规则！核实过：'茂'(140次,"叶茂盛时采收")、
#    '苔'(80次,"苦苣苔科")、'蕾'(575次,"干燥花蕾")、'萼'(666次,"萼筒")、
#    '替'(11次,"被替代对照品") 在正文里都**合法**，整字替换会制造新错误。
#    只有 '幵'、'矶'、'苜' 经全库核对在正文中亦从不合法出现，才允许整字替换；
#    其余一律走**词组级**（如「川萼」「蕾香」「菌麻」这种非法词组）。
#    另有 '砚'：正文中「二甲亚砚」应为"二甲基亚砜"、「绿砚」应为"绿矾"，
#    同一字对应两种正确写法，故只处理「绿砚」这一确定词组。
CONTEXT_FIXES = [
    # ---- 异体字/近形字回流（OCR 把简体字认成繁体或别的字，用户打简体就搜不到）----
    # 逐字核对过全库上下文才敢加，**不要**用 zhconv 整库转繁简：
    # 「癥」在《药典》里是正确用法（化癥回生片），整库转换会把它错改成「症」。
    (r'梔', '栀'),        # 8,892 次！梔子→栀子、茵梔黄→茵栀黄、清火梔麦→清火栀麦
    (r'豬', '豨'),        # 30 次，**全部**是 豨薟草 / 豨桐（正文中「猪」另正确出现 303 次）
    (r'签草', '薟草'),     # 豨签草→豨薟草（「签」在「固定」等语境下合法，故用词组）
    (r'蔥', '蒽'),        # 55 次**全部**是 蒽（总蒽醌 / 游离蒽醌），不是「葱」
    (r'蒽[醍酿醞酝]', '蒽醌'),   # 「醌」本身也被认错：总蒽醍→总蒽醌
    # ⚠️「醍」是多义的（乙醍 1485 次=乙醚、油醍 695 次=石油醚、蔥醍 41 次=蒽醌），
    #    **不能**做整字替换——曾按「醍→滤」整字替换，结果把「合并乙醍液」改成了
    #    「合并乙滤液」（正确应为乙醚液）。故只用下列确定词组规则。
    (r'乙醍', '乙醚'),
    (r'油醍', '油醚'),    # 石油醚
    (r'豨签', '豨薟'),    # 毛梗豨签 → 毛梗豨薟（正文里「簽」写作「签」）
    (r'葦', '苇'),        # 16 次
    (r'癢', '痒'),        # 2 次
    # ---- 整字级（已核对全库正文无合法用法）----
    (r'幵', '开'),        # 展幵→展开、幵胃山楂丸→开胃山楂丸
    (r'矶', '矾'),        # 白矶→白矾、皂矶→皂矾
    (r'苜', '苷'),        # 皂苜→皂苷、柚皮苜→柚皮苷
    # 「澹」一字多义：正文 57 次里 55 次是「溏」（便溏/五更溏泻/大便溏薄/阿胶珠
    #   「内无溏心」）、2 次是「谵」（痰热谵狂/神昏谵语）。对照组：溏 与 谵
    #   在全库正文都是 **0 次**——即两者的每一次出现都被 OCR 认成了「澹」。
    #   必须**先**用词组规则摘走 2 个「谵」，再整字归「溏」；顺序颠倒会把
    #   「谵语」错改成「溏语」。
    (r'澹狂', '谵狂'),    # 清热解毒…痰热谵狂（安宫牛黄类）
    (r'澹语', '谵语'),    # 热入心包，神昏谵语
    (r'澹', '溏'),        # 便澹→便溏、五更澹泻→五更溏泻、大便澹薄→溏薄、
                          # 内无澹心→内无溏心、澹而不爽→溏而不爽（共 55 处）
    (r'洶', '沟'),        # 7 次全是显微描述：孔洶→孔沟、纵洶→纵沟、横洶→横沟
                          # （对照组「沟」375 次合法，洶 从不合法）
    # ================= 缺陷 19：又一批系统性错字（2026-09-23）=================
    # 判据与缺陷 13 相同：**正确写法在全库正文为 0 次**；错字若有合法用法则用词组级。
    # 每个数字都是从 data/processed/chunks.json 全库实测得到的。
    #
    # 「昔」→「苷」的兜底规则见下方「苷类」区块——**必须排在 `X\s*昔` 之后**，
    #   否则「人参皂 昔」会变成「人参皂 苷」（断词空格留在原处）。
    # 「胱腹 / 底腹」→「脘腹」：脘 全库 0 次
    #   ⚠️ 必须词组级——「胱」在「膀胱经」合法（53 次）、「底」在「圆底烧瓶」合法（127 次）
    (r'胱腹', '脘腹'),
    (r'底腹', '脘腹'),
    # 「癥痕」→「癥瘕」：瘕 全库 0 次；痕 在「痕迹/茎痕/须根痕/斑痕」合法（356 次）
    (r'癥痕', '癥瘕'),
    (r'痕积聚', '瘕积聚'),      # 遗留的「】痕积聚」（前面的「癥」已丢失，无法回填）
    # 「瘰疬」：痂 43 次里 41 次该是「疬」（痕 痂/瘪痂/療痂/原痂/瘻痂），**瘰 全库 0 次**；
    #   血痂（2 次）合法 → 先做词组级，再用后行兜底，且用否定断言保护「血痂」
    (r'[痕瘪療原瘻]\s?痂', '瘰疬'),
    (r'瘻', '瘿'),              # 瘻瘤→瘿瘤、瘻疮毒→瘿疮毒（瘿 全库 0 次）
    (r'(?<!血)痂', '疬'),
    # 「衄血」：衄 全库 0 次（吐血、朝血，崩漏下血 / 血热吐朝 / 鼻朝 齿蛻）
    (r'朝\s?血', '衄血'),
    (r'吐朝', '吐衄'),
    (r'鼻朝', '鼻衄'),
    (r'齿蛻', '齿衄'),
    # 「朝藿定 / 碘化铋钾」：藿 在「藿香」合法（214 次）；铋 全库 0 次
    (r'朝蕾定', '朝藿定'),
    (r'碘化朝钾', '碘化铋钾'),
    # 「蛲虫 / 蜣螂 / 甾酮」+「蛻」是「蜕」的异体字（蜕 118 次合法）
    (r'蛻虫病', '蛲虫病'),
    (r'蛻螂', '蜣螂'),
    (r'[蛻蜕]皮螢酮', '蜕皮甾酮'),   # 甾 全库 0 次
    (r'蛻', '蜕'),
    # 「甾醇」：當醇→甾醇（千金子當醇/麦角當醇→甾醇）；β 全库 0 次，「步谷」是 β-谷 的误认
    (r'步谷當醇', 'β-谷甾醇'),
    (r'當醇', '甾醇'),
    # ---- 零星（各 1~2 次，均有上下文互证）----
    (r'0\.\s?1%歸酸', '0.1%磷酸'),   # 乙腈-0.1%磷酸（37:63）作流动相
    (r'三氯化歸', '三氯化铁'),         # 三氯化铁试液（318 次合法写法）
    (r'歸钾', '锑钾'),                 # 酒石酸锑钾
    (r'陳干', '晾干'),                 # 取出，晾干（晾干 5,101 次）
    (r'豨菴草', '豨莶草'),             # 药名（莶 全库 0 次）
    # ---- 二次扫描（把上面修完后再扫剩余形态，又挖出的几处）----
    (r'淫羊蕾', '淫羊藿'),             # **326 次**！正文里「淫羊藿」原本只剩 4 次
                                        # （「花蕾」17 次合法 → 只能词组级）
    (r'胱\s?痞', '脘痞'),              # 湿滞伤中，胱痞吐泻 → 脘痞（脘 全库 0 次）
    (r'瘪疡', '瘰疬'),                 # 瘪 在干瘪/瘦瘪 合法（溃疡 78/疮疡 70 合法）
    (r'療疡', '瘰疬'),
    (r'療病', '瘰疬'),                 # 痈肿療病、療病痰核 → 瘰疬
    # ---- 药名在正文里的残留变体（须先于苷类规则）----
    (r'川萼', '川芎'),
    (r'黄苓', '黄芩'),
    (r'黄芷', '黄芪'),
    (r'广蕾香', '广藿香'),          # 必须先于「蕾香」
    (r'蕾香', '藿香'),
    (r'菌麻', '蓖麻'),
    (r'青曝石', '青礞石'),
    (r'香薫', '香薷'),
    (r'石香鬻', '石香薷'),
    (r'篇蓄', '萹蓄'),
    (r'惹米', '薏苡'),
    (r'马齿竟', '马齿苋'),
    (r'马齿苑', '马齿苋'),
    (r'龙關I叶', '龙脷叶'),
    (r'绿砚', '绿矾'),              # 「二甲亚砚」属另一类错字，保持不动
    # ---- 苷类：各种 OCR 变体 ----
    (r'(皂|总|黄芩|连翘|芍药|柚皮|白头翁皂)\s*昔', r'\1苷'),   # 容忍「人参皂 昔」这类断词
    (r'(皂|总)苔', r'\1苷'),        # 积雪草总苔→积雪草总苷
    (r'苷苷', '苷'),                # 防重复
    # 缺陷 19 兜底：**2,614 次「昔」全是 X苷 的误认**（黄芪甲昔/橙皮昔/栀子昔/糖昔/红景天昔…），
    #   合法用法（昔康/昔洛韦/往昔/古昔/今昔）**全库 0 次**；苷本身另有 3,311 次。
    #   位置必须在 `X\s*昔` 之后，以顺带吃掉断词空格（人参皂 昔→人参皂苷）。
    #   它同时修**检索**：此前查「橙皮苷」匹配不到语料里的「橙皮昔」。
    (r'昔', '苷'),
    # ---- 通则编号与"照"后面的常见错误 ----
    (r'照髙效液相色谱法', '照高效液相色谱法'),
    (r'照高效液相色谱法（通则。512）', '照高效液相色谱法（通则0512）'),
    (r'通则。832', '通则0832'),
    (r'通则。502', '通则0502'),
    (r'通则。512', '通则0512'),
]

# 全角数字 → 半角
FULLWIDTH_DIGIT_MAP = str.maketrans(
    '０１２３４５６７８９',
    '0123456789'
)

# 全角字母 → 半角
FULLWIDTH_ALPHA_MAP = str.maketrans(
    'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
    'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ',
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
)


def clean_text(text: str) -> str:
    """
    清洗单段文本。
    
    清洗步骤:
    1. OCR 章节标记纠错
    2. 上下文敏感的字符纠错
    3. 全角→半角转换（数字和字母）
    4. 多余空白合并
    5. 异常字符清理
    """
    if not text:
        return text

    # 1. OCR 章节标记纠错
    for wrong, correct in SECTION_OCR_FIXES.items():
        text = text.replace(wrong, correct)

    # 2. 上下文敏感的字符纠错
    for pattern, replacement in CONTEXT_FIXES:
        text = re.sub(pattern, replacement, text)

    # 3. 全角→半角
    text = text.translate(FULLWIDTH_DIGIT_MAP)
    text = text.translate(FULLWIDTH_ALPHA_MAP)

    # 4. 多余空白合并
    # 合并连续空格
    text = re.sub(r'[ \t]+', ' ', text)
    # 去除行首行尾空格
    text = '\n'.join(line.strip() for line in text.split('\n'))
    # 合并连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 5. 异常字符清理
    # 去除一些 OCR 残留的特殊符号
    text = text.replace('■', '。')   # OCR 残留的句末标记
    text = text.replace('■', '。')
    # 修复 "0。832" → "0832" （通则编号中的多余句号）
    text = re.sub(r'通则0?\.?(\d{4})', r'通则\1', text)

    return text.strip()


def clean_entry(entry: dict) -> dict:
    """
    清洗一个药品条目（dict格式，来自parser输出的JSON）。
    
    Args:
        entry: 单个药品条目的字典
        
    Returns:
        清洗后的条目字典
    """
    # 清洗药品名 - 关键：去除空白以匹配查询中的"人参" vs "人 参"
    # ⚠️ 必须用 \s 而非只去半角空格：docx 里药名内部可能带 **`<w:tab/>`**
    #    （实例：`策\t巨` → 应为「菊苣」），而 clean_text 的
    #    `re.sub(r'[ \t]+', ' ')` 会把 **Tab 折叠成空格**。若此处只去 ' '，
    #    顺序上就成了"先失去去除机会、后被 clean_text 变成空格并保留"，
    #    于是库里出现 `策 巨` 这种带空格的药名，精确匹配全部失效。
    drug_name = re.sub(r'\s+', '', entry.get('drug_name', ''))
    entry['drug_name'] = clean_text(fix_drug_name(drug_name))

    # 清洗 parent_drug（子剂型的父级药品名）：同样处理
    parent_drug = re.sub(r'\s+', '', entry.get('parent_drug', ''))
    if parent_drug:
        parent_drug = fix_drug_name(parent_drug)
    entry['parent_drug'] = clean_text(parent_drug)
    
    entry['pinyin_name'] = clean_text(entry.get('pinyin_name', ''))
    entry['latin_name'] = clean_text(entry.get('latin_name', ''))
    entry['intro_text'] = clean_text(entry.get('intro_text', ''))

    # 清洗各章节内容
    cleaned_sections = []
    for section in entry.get('sections', []):
        # 先对章节名做规范化纠错
        raw_section_name = section.get('section_name', '')
        normalized_name = SECTION_NAME_NORMALIZE.get(raw_section_name, raw_section_name)
        cleaned_section = {
            'section_name': clean_text(normalized_name),
            'content': clean_text(section.get('content', '')),
            'table_markdown': clean_text(section.get('table_markdown', '')) if section.get('table_markdown') else None,
            'raw_paragraphs': [clean_text(p) for p in section.get('raw_paragraphs', [])],
        }
        # 跳过内容为空且无表格的章节
        if not cleaned_section['content'] and not cleaned_section['table_markdown']:
            continue
        cleaned_sections.append(cleaned_section)

    entry['sections'] = cleaned_sections
    return entry


# ============================================================
# 跨药污染尾巴剪裁
# ============================================================
# 现象（2026-09-22 量化）：部分条目的某一节末尾**粘着下一个药的开头**，
# 例：`苍术-饮片` 的【贮藏】= "置通风干燥处。\nCang'erzi\nXANTHII FRUCTUS\n本品为菊科植物苍耳…"
#      `茵栀黄胶囊` 的【含量测定】里连吞 1787 字的别的药内容。
# 根因与缺陷 11 同源：源 docx 是多栏/表格版面（<w:cols w:num="2"> 943 处、<w:tbl> 996 处），
# 边界处的药名行缺失或被拆散，解析器无法据此切分。
#
# ⚠️ 这里**只能保守剪裁**，不能按"仓储后还有章节"之类的结构规则一刀切：
#   ① 药典允许【贮藏】之后跟 `附：xxx质量标准` / `注：…`（实测最长 2032 字**是合法的**），
#      所以不能按句号或长度剪；
#   ② 中文正文里也可能出现独占一行的西文（如含量测定里的公式 `bOO X w`），
#      单看"有拼音样的一行"会误伤（实测 `滑石粉` 就被误判过）。
# 故判定要求**两个信号同时成立**：尾部以 `本品为`（药典条目正文的固定起头）开头，
# 或"独占一行的拼音/拉丁名 + 其后 250 字内出现 `本品为`"。
# 而且若前面已出现 `附：/注：/【制剂】` 这类**合法续接标记**，一律不动。

# 合法续接标记：附（附加质量标准）、注（脚注）、制剂
_LEGIT_TAIL_MARK = re.compile(r"(附\s*[:：]|注\s*[:：]|[\[【〔(]\s*[:：]?\s*制\s*剂)")
# 独占一行的拼音/拉丁名（如 `Sanqi`、`CURCUMAE LONGAE RHIZOMA`）
_ASCII_NAME_LINE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 .'\-]{3,}\s*$")
# 药典条目正文的固定起头
_INTRO_HEAD = "本品为"

# 【贮藏】之后唯一合法的章节
_SECTIONS_ALLOWED_AFTER_STORAGE = {"制剂"}


# 【贮藏】这一节可用的**更强规则**：首行是完整陈述句时，其后只允许 附/注/制剂。
# 依据：仓储正文本身就是一两句短话；实测 89 个"过长仓储"里绝大多数其后跟的是
# 别的药的鉴别/含量测定碎片（`对照品溶液的制备…`、`密称定，置具塞锥形瓶中…`），
# 这类碎片**不含 `本品为`**，通用规则抓不到。
# 唯一的合法例外是"多行仓储"（如 `野菊花栓` 的两种基质分别陈列表述），
# 其特征是**首行不是完整句子**（不以 。／； 结尾），据此排除。
_COMPLETE_SENTENCE_END = ("。", "；")
# 单行仓储里，第一个句号之后超过这个长度且非合法续接 → 判为污染
_SINGLE_LINE_TAIL_MIN = 15


def _cut_storage_tail(text: str) -> int:
    """【贮藏】专用：首行是完整陈述句时，其后只允许 附/注/制剂 —— 否则截断。

    Returns: 截断位置；-1 表示保留
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        # 单行也要查：污染可能接在**同一行**的空格之后
        # （实例 `玄参`："置干燥处，防霉，防蛀。 于同一硅胶G薄层板上，以三氯甲烷-甲醇…"）
        m = re.search(r"[。]", text)
        if not m:
            return -1
        rest = text[m.end():].strip()
        if len(rest) > _SINGLE_LINE_TAIL_MIN and not _LEGIT_TAIL_MARK.match(rest):
            return m.end()
        return -1
    if not lines[0].rstrip().endswith(_COMPLETE_SENTENCE_END):
        return -1                       # 多行仓储（各基质分别陈述）→ 合法，保留
    rest = "\n".join(lines[1:]).strip()
    if _LEGIT_TAIL_MARK.match(rest):
        return -1                       # 附：/注：/制剂 → 合法续接
    for i, ln in enumerate(text.split("\n")):
        if ln.strip():
            return len(text.split("\n")[0])
    return -1


def find_cross_drug_cut(text: str, section_name: str = "") -> int:
    """在正文里找出「下一个药从哪里开始」的位置；找不到返回 -1。

    Args:
        text: 某一节的正文
        section_name: 章节名（`贮藏` 会额外启用更强规则，见上方注释）

    Returns:
        应截断的位置（保留 `[:cut]`）；-1 表示判定为无跨药污染
    """
    if not text or len(text) < 25:
        return -1
    if section_name == "贮藏":
        cut = _cut_storage_tail(text)
        if cut >= 0:
            return cut
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if i == 0:
            continue
        if not ln.strip():
            continue
        head = "\n".join(lines[:i])
        if _LEGIT_TAIL_MARK.search(head):
            return -1                       # 前面已有合法续接标记 → 整段保留
        if ln.strip().startswith(_INTRO_HEAD):
            return len(head)
        if _ASCII_NAME_LINE.match(ln) and _INTRO_HEAD in text[len(head):len(head) + 250]:
            return len(head)
    return -1


def trim_cross_drug_tails(entries: list) -> dict:
    """剪掉条目里粘着的「下一个药的开头」，并丢掉【贮藏】之后不该有的章节。

    **只做减法、不做猜测**：被判为污染的部分直接丢弃（它本来就挂错了条目），
    统计信息返回给调用方打印，便于人工复核到底丢了什么。

    Returns:
        {"entries_touched": n, "chars_dropped": m, "sections_dropped": k, "samples": [...]}
    """
    touched = dropped_chars = dropped_sections = 0
    samples = []
    for e in entries:
        hit = False
        for s in e["sections"]:
            content = s.get("content") or ""
            cut = find_cross_drug_cut(content, s["section_name"])
            if cut > 0:
                dropped_chars += len(content) - cut
                if len(samples) < 8:
                    samples.append((e["drug_name"], s["section_name"],
                                    content[cut:cut + 40].replace("\n", "⏎")))
                s["content"] = content[:cut].rstrip()
                hit = True

        # 【贮藏】之后只允许【制剂】（药典编排如此）；其余章节属跨药污染
        secs = e["sections"]
        idxs = [i for i, s in enumerate(secs) if s["section_name"] == "贮藏"]
        if idxs:
            last = idxs[-1]
            keep, drop = [], []
            for j, s in enumerate(secs):
                if j > last and s["section_name"] not in _SECTIONS_ALLOWED_AFTER_STORAGE:
                    drop.append(s)
                else:
                    keep.append(s)
            if drop:
                dropped_sections += len(drop)
                e["sections"] = keep
                hit = True
        if hit:
            touched += 1
    return {"entries_touched": touched, "chars_dropped": dropped_chars,
            "sections_dropped": dropped_sections, "samples": samples}


def merge_same_name_entries(entries: list) -> list:
    """合并**同名**条目（并集去重，不丢任何内容）。

    来源：原 docx 用的是多栏/表格版面（`<w:cols w:num="2">` 943 处、`<w:tbl>` 996 处），
    个别页面上两个药的名字块按**表格行**排列（`[鹅不食草|筋骨草]`、`[Ebushicao|Jingucao]`、
    `[CENTIPEDAE HERBA|AJUGAE HERBA]`），线性读取后同一个饮片会被拆成两条
    （实测 7 组：刀豆-饮片、小蓟-饮片、苍术-饮片、禹州漏芦-饮片、锁阳-饮片、
    豨薟草-饮片、稻芽-饮片）。同名条目会让候选池里出现重复条目、稀释排序。

    做法：同名条目**按章节 (section_name, content) 去重后取并集**，顺序按首次出现；
    顺带补齐缺失的拼音/拉丁名/概述。**不删除任何内容**——残留的跨药尾部文字
    属另一个更底层的问题，需另行处理。
    """
    by_name, order = {}, []
    for e in entries:
        n = e["drug_name"]
        if n not in by_name:
            by_name[n] = e
            order.append(n)
            continue
        base = by_name[n]
        seen = {(s["section_name"], s.get("content") or "") for s in base["sections"]}
        for s in e["sections"]:
            key = (s["section_name"], s.get("content") or "")
            if key not in seen:
                base["sections"].append(s)
                seen.add(key)
        for f in ("pinyin_name", "latin_name", "intro_text"):
            if not (base.get(f) or "").strip() and (e.get(f) or "").strip():
                base[f] = e[f]
    return [by_name[n] for n in order]


def clean_entries(entries: list) -> list:
    """批量清洗药品条目"""
    cleaned = []
    skipped = 0
    for entry in entries:
        entry = clean_entry(entry)
        # 跳过完全没有内容的条目
        if not entry['sections'] and not entry['intro_text']:
            skipped += 1
            continue
        cleaned.append(entry)

    # 剪掉跨药污染尾巴 + 丢掉【贮藏】之后不该有的章节（缺陷 11 的残留，见函数注释）
    tail_stats = trim_cross_drug_tails(cleaned)

    before = len(cleaned)
    cleaned = merge_same_name_entries(cleaned)
    merged = before - len(cleaned)

    print(f"清洗完成: {len(entries)} → {len(cleaned)} 条目 (跳过空条目 {skipped} 个, 合并同名 {merged} 个)")
    if tail_stats["entries_touched"]:
        print(f"  跨药尾巴剪裁: {tail_stats['entries_touched']} 个条目, "
              f"丢弃 {tail_stats['chars_dropped']} 字 / {tail_stats['sections_dropped']} 个越界章节")
        for name, sec, prev in tail_stats["samples"]:
            print(f"    · {name} 【{sec}】 起于: {prev!r}")
    return cleaned


# ============================================================
# 命令行入口
# ============================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import DRUGS_JSON_PATH

    print("=" * 60)
    print("数据清洗器")
    print("=" * 60)

    # 读取解析后的数据
    with open(DRUGS_JSON_PATH, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    print(f"读取 {len(entries)} 个药品条目")

    # 清洗
    cleaned_entries = clean_entries(entries)

    # 保存（覆盖原文件）
    with open(DRUGS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(cleaned_entries, f, ensure_ascii=False, indent=2)

    print(f"已保存清洗后数据到 {DRUGS_JSON_PATH}")

    # 统计清洗效果
    print(f"\n清洗效果统计:")
    
    # 检查"皂昔"→"皂苷"替换
    saponin_fixes = 0
    for e in cleaned_entries:
        for s in e.get('sections', []):
            content = s.get('content', '')
            if '皂苷' in content:
                # 检查是否还有残留的"皂昔"
                pass
        # 这里只是展示，实际替换已在clean_text中完成
    
    # 检查章节标记纠错
    section_names = set()
    for e in cleaned_entries:
        for s in e.get('sections', []):
            section_names.add(s['section_name'])
    print(f"  章节类型: {sorted(section_names)}")
    
    # 检查是否还有已知的OCR错误
    remaining_errors = 0
    for e in cleaned_entries:
        all_text = json.dumps(e, ensure_ascii=False)
        for wrong in ['検查', '測定', '礎藏', '疋藏', '皂昔']:
            if wrong in all_text:
                remaining_errors += all_text.count(wrong)
    print(f"  残留已知OCR错误: {remaining_errors} 处")

    # 样例
    print(f"\n{'='*60}")
    print("清洗后样例:")
    print(f"{'='*60}")
    for e in cleaned_entries:
        if '人参' in e['drug_name'] and not e.get('is_yinpian'):
            print(f"\n--- {e['drug_name']} ---")
            for s in e.get('sections', [])[:5]:
                print(f"  【{s['section_name']}】 {s['content'][:100]}...")
            break
