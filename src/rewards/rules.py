"""确定性规则层：两个判定函数，零噪声、可逐条核对。

本模块在整条链路里的位置：数据筛选、在线奖励、离线评测三处共用的硬门。它的输出没有随机性，
同一个输入永远同一个结果，所以它既能当训练时的硬门，也能当评测时的客观指标。

两个职责，就这两个：

1. :func:`detect_rag_style` —— 这段 think 里有没有"我在查资料"的**表面标记**。
2. :func:`answer_in_v1_pool` —— 这个答案的结论极性和关键数字，还在不在基座自己认可的范围里。

**规则刻意不碰"是不是换词复述照抄"。** 一个更省事的做法是让规则顺手也判照抄：
凡是引用了法规名或文号的句子就记一笔痕迹。这条路的问题不是"抓得不够准"，而是它会
把嵌进推理里的合法引用（"按《XX》第几条，这里适用 3%"）判成痕迹——而这类引用恰恰
让答案更准。硬按这个信号优化，模型学到的是"别引用法条"，最终 grounding 塌掉、
答案开始飘。所以照抄判断整个交给裁判模型，规则只干上面两件确定性的活。

**这层挡不住什么**：换词复述。参考资料里写"月销售额未超过10万元的小规模纳税人免征增值税"，
模型写成"小规模纳税人这一档，月销售额不到10万的免增值税"——一个检索腔词都没有，
规则完全看不见，骨子里还是逐条搬运。那是裁判模型的活，见 :mod:`src.rewards.judge`。
"""

from __future__ import annotations

import re
import unicodedata

# ===========================================================================
# 规则一：检索腔的表面标记
# ===========================================================================
# 四类，每类都是逐字命中、确定性的。分类不是为了好看，是为了出问题时能定位到
# 具体哪一类误伤——比如 D 类图床链接曾经和文件名后缀重叠计了两次，靠分类才看得出来。

_RAG_PATTERNS = {
    "A_检索装置腔": [
        r"参考问答对", r"参考资料", r"参考内容", r"检索结果", r"资料(显示|表明|指出)",
        r"原始(问答对|资料|回答)", r"根据(检索|参考|提供的资料|上述资料|以上资料)",
        r"如(上|前)(文|述|参考|资料|所示)", r"对照(参考|资料|问答)",
    ],
    "B_编号引用": [
        r"问答对\s*[0-9一二三四五六七八九十]", r"问题\s*\d+\s*[:：]", r"回答\s*[:：]",
        r"第\s*[0-9一二三四五六七八九十]\s*个?参考", r"(逐条|依次|分别)(对照|参考|归纳)",
    ],
    "C_客服话术": [
        r"小贴士", r"温馨提(示|醒)", r"参考下图", r"如下图", r"您可以参考", r"哦[~～]", r"亲[，,~～]",
    ],
    "D_图床附件": [
        r"<img\b", r"https?://\S*(?:aliyuncs|oss-|servu)\S*", r"\b\S+\.(?:png|jpe?g|xlsx?|pdf)(?![a-z0-9])",
    ],
}

# D 类大小写无关：截图文件名常见 .PNG/.JPG、<IMG>。
# 尾部用负向预查而不是 \b，是为了让紧贴中文的文件名（"…table1.png的数据"）也能命中——
# \b 在中文和字母之间不成立，会漏掉这一整类。
_RAG_RE = {
    name: [re.compile(p, re.I) if name == "D_图床附件" else re.compile(p) for p in patterns]
    for name, patterns in _RAG_PATTERNS.items()
}

# 清单式甩文号：唯一被算作痕迹的"政策引用"情形。
# 三个分支分别抓：《法规名》第X条、财税〔2019〕13号这种旧式文号、公告2023年第1号这种现行公告体。
_DOC_TOKEN_RE = re.compile(
    r"(?:《[^》]{2,80}》(?:第[一二三四五六七八九十百千万\d]+条)?|"
    r"[^\s，。；、]{0,12}[财税会发函公告令]\s*[〔\[\(（]?\d{4}[〕\]\)）]?\s*\d+\s*号|"
    r"[^\s，。；、]{0,16}(?:公告|令)\s*(?:\d{4}\s*年)?\s*第?\s*\d+\s*号)"
)
_POLICY_LABEL_RE = re.compile(r"(?m)^\s*(?:政策依据|参考文件|参考法规|文件依据)\s*[:：]")

# 按句豁免用的三个正则：表单名不算文号、句子里有推理词就算"服务推理"。
_FORM_NAME_RE = re.compile(r"《[^》]{1,80}(?:表|单|凭证)》")
_CONTENT_WORD_RE = re.compile(
    r"(规定|按照|明确|适用|应当|可以|不得|免征|征收|处理|确认|扣除|申报|"
    r"判断|区分|属于|不属于|满足|导致|计算|缴纳|计入|结转|抵扣|"
    r"依据|由于|若|如果|因此|所以|意味着|对应|决定|需要|无需|涉及|"
    r"享受|选择|采用|执行|发生|取得|提供|销售|转让|出租|支付|收到)"
)
_SENT_SPLIT_RE = re.compile(r"[。！？!?\n;；]+")

# 一句话里出现几个文号才算"在甩清单"。设 3 是因为 1~2 个通常是正常引用；
# 到 3 个基本就是把政策依据列表照搬过来了。
_DOCS_PER_SENTENCE_LIMIT = 3
# 报告里最多列几条甩号样例。只是给人看的，不影响判定。
_MAX_REPORTED_DOCS = 8


def _split_sentences(text: str) -> list[str]:
    """按中文句末标点切句。按句判定是"合法引用豁免"的前提。"""
    return [s.strip() for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]


def detect_rag_style(think: str) -> dict:
    """判这段 think 有没有检索腔的表面标记。

    :param think: 推理段原文
    :return: ``{has_rag_style, spans, n_by_type, n}``。`spans` 是命中的片段和类别，
             出问题时可以逐条核对到底哪个词被算成了痕迹。
    """
    text = think or ""
    spans: list[dict] = []
    n_by_type: dict[str, int] = {}

    # 1.【逐字命中】A/B/C/D 四类各自扫一遍，收下起止位置以便去重
    raw: list[tuple[int, int, str, str]] = []
    for type_name, regexes in _RAG_RE.items():
        for regex in regexes:
            for match in regex.finditer(text):
                raw.append((match.start(), match.end(), match.group(0), type_name))

    # 1.1 同类同位置去重：同一个词可能被同类里两条正则各抓一次
    seen: set = set()
    unique: list[tuple[int, int, str, str]] = []
    for start, end, matched, type_name in sorted(raw, key=lambda x: (x[0], x[1])):
        if (start, end, type_name) in seen:
            continue
        seen.add((start, end, type_name))
        unique.append((start, end, matched, type_name))

    # 1.2 被同类更大片段完全包住的丢掉：一条图床链接本身就含 .png 后缀，
    #     不去重会把同一处痕迹数成两笔，痕迹计数虚高。
    for i, (start, end, matched, type_name) in enumerate(unique):
        contained = any(
            j != i and other_start <= start and end <= other_end
            and (other_end - other_start) > (end - start) and other_type == type_name
            for j, (other_start, other_end, _text, other_type) in enumerate(unique))
        if contained:
            continue
        spans.append({"text": matched, "type": type_name})
        n_by_type[type_name] = n_by_type.get(type_name, 0) + 1

    # 2.【按句豁免】文号只有"在甩清单"时才算痕迹
    #    判据：这一句里没有任何推理词（说明它不在推进判断，只是在列依据），或者一句里塞了 3 个以上文号。
    standalone_docs: list[str] = []
    for sentence in _split_sentences(text):
        reduced = _FORM_NAME_RE.sub("", sentence)     # 《XX表》《XX凭证》是表单名，不当文号
        docs = _DOC_TOKEN_RE.findall(reduced)
        if not docs:
            continue
        if not _CONTENT_WORD_RE.search(reduced) or len(docs) >= _DOCS_PER_SENTENCE_LIMIT:
            standalone_docs.extend(docs)

    label_match = _POLICY_LABEL_RE.search(text)
    if len(standalone_docs) >= _DOCS_PER_SENTENCE_LIMIT or label_match:
        for doc in standalone_docs[:_MAX_REPORTED_DOCS]:
            spans.append({"text": doc, "type": "D_清单式甩文号"})
        if label_match and not standalone_docs:
            # 只有"政策依据："这个标签、后面的文号格式没被正则抓到时，标签本身就是充分证据
            spans.append({"text": label_match.group(0).strip(), "type": "D_清单式甩文号"})
        n_by_type["D_清单式甩文号"] = max(len(standalone_docs), 1 if label_match else 0)

    return {"has_rag_style": bool(spans), "spans": spans, "n_by_type": n_by_type, "n": len(spans)}


# ===========================================================================
# 规则二：答案还在不在基座认可池里
# ===========================================================================
# 这一条判的是"漂没漂"，不是"对不对"。基座答错了，模型跟着答错也算没漂——
# 因为这条链路的目标是只改 think 的表达方式，答案本身不许动。
# 把它当"准确率"来读是误读，见 README 里指标那一节。

def nfkc(text: str) -> str:
    """全角半角归一。不做这一步，"１０万" 和 "10万" 会被当成两个不同的数字。"""
    return unicodedata.normalize("NFKC", text or "")


# 关键数字按类型分别抽。日期单独一类，因为它要按"粒度前缀"判覆盖，其余按集合比对。
_PCT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")                                    # 13% / 5%
_FRAC_RE = re.compile(r"万分之[零一二三四五六七八九十百点\d]+|千分之[零一二三四五六七八九十百点\d]+")
_MONEY_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:万元|亿元|万|亿|元)")          # 10万元 / 10万
_DATE_RE = re.compile(r"\d{4}\s*年(?:\s*\d{1,2}\s*月)?(?:\s*\d{1,2}\s*日)?")   # 2023年 / 2023年1月1日
_DUR_RE = re.compile(r"\d+\s*(?:个)?(?:日|天|个月|月|年|季度)")                # 30天 / 6年

# 结论极性词。相反极性 = 漂了。
# 长词排前面并且命中后从文本里消位，否则 "不超过" 会被拆出一个 "超过"，
# 一句话同时带上正反两个极性，判定全错。
_POLARITY = ["免征", "免税", "不征税", "应缴", "应纳税", "需缴纳", "不得", "不可以", "无需", "不需要",
             "可以", "允许", "禁止", "超过", "不超过", "属于", "不属于"]
_POLARITY_SORTED = sorted(_POLARITY, key=len, reverse=True)


def _norm_money(item: str) -> str:
    """金额单位归一：万元→万、亿元→亿。不归一，"10万" 和 "10万元" 会被判成两个不同的数字。"""
    return item.replace("万元", "万").replace("亿元", "亿")


def _polarity_set(text: str) -> set[str]:
    """抽极性词。长词优先命中，命中后用全角空格占位，短子串就不会被重复抽出。"""
    remaining = text
    found: set[str] = set()
    for word in _POLARITY_SORTED:
        if word in remaining:
            found.add(word)
            remaining = remaining.replace(word, "　")
    return found


def extract_facts(text: str) -> dict:
    """从一段文本里抽出【极性 / 数字 / 日期】三类事实并归一化。

    日期先抽并从文本里消位，否则期限正则会从 "2023年1月1日" 里再抽出 "1月"、"1日"
    两个根本不存在的期限。

    :param text: 任意文本（答案、推理、题面都行）
    :return: ``{polarity, value, date}``，三个字符串集合
    """
    normalized = nfkc(text)
    polarity = _polarity_set(normalized)

    dates: set[str] = set()
    consumed = normalized
    for match in _DATE_RE.findall(normalized):
        dates.add(re.sub(r"\s+", "", match))
        consumed = consumed.replace(match, " ", 1)

    values: set[str] = set()
    for regex in (_PCT_RE, _FRAC_RE, _MONEY_RE, _DUR_RE):
        for match in regex.findall(consumed):
            values.add(_norm_money(re.sub(r"\s+", "", match).replace(",", "")))
    return {"polarity": polarity, "value": values, "date": dates}


def _date_covered(date: str, pool_dates: set[str]) -> bool:
    """日期按粒度前缀判覆盖。

    同一个日期写粗写细（"2023年" 和 "2023年1月1日"）不算漂——模型把日期说得更具体
    或更笼统，都不构成"改了答案"。

    :param date: 模型答案里的一个日期
    :param pool_dates: 池子里出现过的全部日期
    :return: 覆盖得上为 True
    """
    return any(date == p or date.startswith(p) or p.startswith(date) for p in pool_dates)


def answer_in_v1_pool(model_answer: str, pool_answers: list[str]) -> dict:
    """判模型答案的极性和关键数字，是不是都被基座认可池覆盖住了。

    池子是基座对同一道题贪心 1 次 + 采样若干次得到的答案集合，代表"基座自己愿意这么答"
    的范围。模型答案里抽出的每个极性、数字、日期都在池里 → 没漂；冒出池子里没有的东西 → 漂了。

    **`comparable` 这个字段单独存在，是为了不把"测不到"伪装成"通过"。** 一条不含任何数字、
    极性的空话答案（"建议结合实际情况处理"），抽出来是空集合，而空集合是任何集合的子集——
    不加这个字段，它会被判成"没漂移"，指标就虚高了。评测里这类样本仍按旧口径计入主指标
    以保持历史可比，但会在 `comparable` 上单列出来审计；在线奖励则直接把它按不合格处理，
    否则模型会学到"少说具体数字就能拿分"。

    :param model_answer: 被评的模型答案
    :param pool_answers: 基座对该题的多条答案
    :return: ``{in_pool, comparable, reason, drift_facts, model_facts, pool_facts, pool_size}``
    """
    # 1.【建池】把池里所有答案的事实并起来
    pool = {"polarity": set(), "value": set(), "date": set()}
    for answer in pool_answers or []:
        if not (answer or "").strip():
            continue
        facts = extract_facts(answer)
        pool["polarity"] |= facts["polarity"]
        pool["value"] |= facts["value"]
        pool["date"] |= facts["date"]

    model = extract_facts(model_answer)
    model_has_facts = bool(model["polarity"] or model["value"] or model["date"])
    pool_has_facts = bool(pool["polarity"] or pool["value"] or pool["date"])

    # 2.【可比性】先判这道题到底测不测得了，再判过没过
    empty_answer = not (model_answer or "").strip()
    if empty_answer:
        # 子分支A1：真空输出。它不是"没漂移"——空集合是任何池的子集，不特判会白送一个通过。
        comparable, reason = True, "empty_answer"
    elif not model_has_facts:
        # 子分支A2：非空但没有可抽的槽位。规则测不到，明确退出可比较分母。
        comparable = False
        reason = "no_comparable_facts" if pool_has_facts else "no_facts_in_either"
    elif not pool_has_facts:
        # 子分支A3：池子本身就没有事实，没有靶子可比。
        comparable, reason = False, "v1_pool_has_no_facts"
    else:
        comparable, reason = True, "ok"

    # 3.【比对】模型多说的极性和数字，加上粒度对不上的日期，就是漂移证据
    drift = sorted(model["polarity"] - pool["polarity"]) + sorted(model["value"] - pool["value"])
    drift += [d for d in sorted(model["date"]) if not _date_covered(d, pool["date"])]

    in_pool = not empty_answer and not drift
    if comparable and reason == "ok" and not in_pool:
        reason = "introduced_new_fact"

    return {
        "in_pool": bool(in_pool),
        "comparable": bool(comparable),
        "reason": reason,
        "drift_facts": sorted(drift),
        "model_facts": sorted(model["polarity"] | model["value"] | model["date"]),
        "pool_facts": sorted(pool["polarity"] | pool["value"] | pool["date"]),
        "pool_size": len(pool["polarity"] | pool["value"] | pool["date"]),
    }
