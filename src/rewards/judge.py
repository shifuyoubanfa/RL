"""裁判打分：给一段 think 打 0~10 的"干净分"，越高越没有换词复述照抄。

本模块在整条链路里的位置：规则层管不到的那一半。规则只能抓表面标记（"参考问答对1"
这种字眼），抓不到换词复述——参考资料里说"月销售额未超过10万元的免征增值税"，
模型写成"这一档月销售额不到10万的免增值税"，一个检索腔词都没有，规则完全看不见，
可它骨子里仍然是在逐条搬运参考。这件事只有大模型判得了，所以整条链路里**唯一**
交给大模型的判断就是它。

三档 k 值，各有各的用途，不许互相冒充：

- ``k_screen``（粗筛）：先打两遍，均值都没比对照高就直接弃，省下贵的那一步。
- ``k_select``（选样）：必须 ≥ 16。:mod:`src.rewards.calibration` 那张 σ 表是按每条打
  16 遍量出来的档内标准差；少打，裁判自身的噪声没被 √16 压下去，σ-可分判据会偏松，
  选进来的样本里混着一批"只是这次抖高了"的。
- ``k_eval``（整体评测）：3 遍。整体评测只要平均分，而平均分的瓶颈是题与题之间的差异，
  加 k 几乎不动标准误（见 :func:`src.rewards.calibration.eval_se`），加了纯烧钱。

打分提示词和标定时用的**必须是同一套**。换了提示词，标定表就不作数，σ-可分判据也就没了依据。
"""

from __future__ import annotations

import contextlib
import json
import re
import statistics

from src.models import judge_client
from src.rewards import budget

# 喂给裁判的参考资料和 think 各自截断到这个长度。
# 目的是把单次请求的输入 token 控制住：判分要打成千上万次，输入长度直接决定总花费。
# 实际语料里超出这个长度的属于少数，截断影响的是尾部细节，不影响"有没有在复述"这个判断。
_REFERENCE_CHAR_LIMIT = 3500
_THINK_CHAR_LIMIT = 4000
# 判分只要一小段 JSON，给足即可。
_MAX_TOKENS = 400

CLEAN_SCORE_SYSTEM = (
    "你是税务 think 的『换词复述照抄』程度评分员。给整段 think 打一个 0-10 的『干净分』："
    "完全没有换词复述照抄=10 分，整段几乎都是把参考逐条换词复述=0 分。"
    "不评文风长短、不判答案对错，只看『有多少是把参考某段换个说法复述了一遍』。只输出 JSON。"
)

# 六个锚点和标定表的六档一一对应。这不是巧合：标定时造的就是这六档样本，
# 提示词里把档位写死，裁判的输出才落在标定出来的那条尺子上。
CLEAN_SCORE_TEMPLATE = """给你一道税务题的【参考资料】和一段【think】。请给这段 think 打一个 0-10 的『干净分』——
分数越高 = 越没有"把参考资料某段换个说法、重新复述一遍"(换词复述照抄)；分数越低 = 越多句子是在逐条复述参考。

评分锚点(共 6 档，照抄越多分越低)：
- 10 分：完全没照抄——全是从问题自己一步步推，或只是把参考里的事实(税率/金额/期限)自然用进推理(这【不算】照抄)。
- 8 分：只有 1 句是把参考某段换词复述。
- 6 分：有 2 句换词复述。
- 4 分：有 3 句换词复述。
- 2 分：有 4 句或更多换词复述。
- 0 分：整段几乎每句都是把参考逐条换词复述(完全照抄)。

关键：把参考里的某个事实自然用在自己的推理里【不算照抄】(例如"这档线是10万，本月9万还在线内")；只有"整句在复述参考某段"才算。
不要因为 think 篇幅长、用词正式、含专业术语就扣分。

【参考资料】
{reference}

【think】
{think}

只输出如下 JSON：{{"clean_score": 0到10之间的数字, "n_copied_est": 你估计的换词复述句数, "reason":"一句话理由"}}"""


def _parse_clean_score(raw: str) -> dict:
    """从裁判的回复里抠出干净分。

    两级兜底：先按整段 JSON 解析；解析不出再用正则单抠 `clean_score` 那个数——
    模型偶尔会在 JSON 前后多写一句话，为这个丢掉一次调用不值得。两级都失败就抛异常，
    上层记一次失败，绝不返回一个编出来的分数。

    :param raw: 裁判返回的原始文本
    :return: ``{clean_score, n_copied_est, reason}``，分数已裁进 [0, 10]
    :raise ValueError: 两级解析都拿不到分数
    """
    text = (raw or "").strip()
    obj = None
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:                   # noqa: PERF203 - 落到下面的正则兜底
            obj = None
    if not isinstance(obj, dict) or obj.get("clean_score") is None:
        loose = re.search(r'"clean_score"\s*:\s*([0-9.]+)', text)
        if not loose:
            raise ValueError(f"干净分解析失败: {text[:160]}")
        obj = {"clean_score": float(loose.group(1))}
    score = max(0.0, min(10.0, float(obj["clean_score"])))
    return {"clean_score": score, "n_copied_est": obj.get("n_copied_est"), "reason": obj.get("reason", "")}


def judge_clean_score(reference: str, think: str, *, temperature: float = 0.0) -> dict:
    """打一遍分。这是最底层的单次调用，上面三档 k 都建在它之上。

    :param reference: 该题的参考资料
    :param think: 被评的推理段
    :param temperature: 判分温度，默认 0 保可复现
    :return: ``{clean_score, n_copied_est, reason, raw}``
    :raise ValueError: 回复解析不出分数
    :raise RuntimeError: 裁判服务调用失败
    """
    prompt = CLEAN_SCORE_TEMPLATE.format(
        reference=(reference or "")[:_REFERENCE_CHAR_LIMIT],
        think=(think or "")[:_THINK_CHAR_LIMIT])
    raw = judge_client.chat(
        [{"role": "system", "content": CLEAN_SCORE_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=temperature, top_p=0.7, max_tokens=_MAX_TOKENS)
    result = _parse_clean_score(raw)
    result["raw"] = raw
    return result


def score_think(reference: str, think: str, k: int) -> dict:
    """打 k 遍取平均。带缓存。

    **全 k 遍都失败时返回 `clean_score=None`、`n=0`，不返回 0.0。** 这一点很要紧：
    0.0 恰好是标定表里"整段照抄"那一档的均值，一次服务故障会被当成一条极脏样本混进
    评测平均里，把整体分数拖下去。调用方必须按 `n > 0` 过滤。

    :param reference: 该题的参考资料
    :param think: 被评的推理段
    :param k: 打几遍
    :return: ``{clean_score, sd, n}``；全失败时 ``{None, 0.0, 0}``
    """
    cached = budget.cache_get(reference, think, k)
    if cached is not None:
        return cached

    scores: list[float] = []
    for _ in range(max(1, k)):
        # 单遍失败降级成"少打一遍"，不是失败。真正全失败的情况由下面的 n==0 分支处理
        with contextlib.suppress(Exception):
            scores.append(float(judge_clean_score(reference, think)["clean_score"]))
    if not scores:
        return {"clean_score": None, "sd": 0.0, "n": 0}

    result = {
        "clean_score": sum(scores) / len(scores),
        "sd": statistics.stdev(scores) if len(scores) >= 2 else 0.0,
        "n": len(scores),
    }
    budget.cache_put(reference, think, k, result)
    return result


def score_for_select(reference: str, think: str, k: int) -> dict:
    """选样口径。断言 k ≥ 16。

    这个断言不是洁癖：σ 标定表量的是每条打 16 遍时的档内标准差，
    :func:`src.rewards.calibration.confident_cleaner` 直接套那张表。k 小了，
    裁判自身的噪声没被 √k 压掉，判据实际比声称的松，而日志上完全看不出来。

    :param reference: 该题的参考资料
    :param think: 被评的推理段
    :param k: 打几遍，必须 ≥ 16
    :return: 同 :func:`score_think`
    :raise AssertionError: k < 16
    """
    assert k >= 16, f"选样必须 k>=16（套 σ 标定表的前提），当前 k={k}"
    return score_think(reference, think, k)


def cleaner_scores(reference: str, clean_think: str, dirty_think: str,
                   *, k_screen: int, k_select: int) -> tuple[float | None, float | None]:
    """两段式打分：先粗筛，过了再上选样口径双评。

    粗筛这一步省的是钱：候选里有相当一部分连均值都没比对照高，压根不用花 16 遍去确认。

    :param reference: 该题的参考资料
    :param clean_think: 候选（希望更干净的那条）
    :param dirty_think: 对照（基座原推理）
    :param k_screen: 粗筛打几遍
    :param k_select: 选样打几遍
    :return: ``(候选选样分, 对照选样分)``；被粗筛掉或打分失败时返回 ``(None, None)``
    """
    screen_clean = score_think(reference, clean_think, k_screen)
    screen_dirty = score_think(reference, dirty_think, k_screen)
    if (screen_clean["clean_score"] is None or screen_dirty["clean_score"] is None
            or screen_clean["clean_score"] <= screen_dirty["clean_score"]):
        return None, None
    return (score_for_select(reference, clean_think, k_select)["clean_score"],
            score_for_select(reference, dirty_think, k_select)["clean_score"])
