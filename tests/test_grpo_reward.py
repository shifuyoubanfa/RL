"""在线奖励函数的契约测试：词典序不许被改掉。

这是整个仓库里最要紧的一组用例。奖励函数一旦从"硬门"退化成"加权和"，
训练会静默走向 reward hacking：三个指标里两个涨、一个慢慢跌，曲线看着一切正常，
直到有人去读输出才发现答案变了。

这些用例全是**离线用例**，不调裁判服务（`use_judge=False`），只验硬门的相对顺序。
"""

from __future__ import annotations

import json

from src.rewards.grpo_reward import RuleWarmupReward, reward_one

POOL = [
    "本月销售额9万元，未超过10万元，可以免征增值税。",
    "销售额9万元低于10万元标准，因此免征增值税。",
]
USER_PROMPT = "【参考问答对】小规模纳税人月销售额未超过10万元的免征增值税。【问题】本月9万元是否免税"


def wrap(think: str, answer: str) -> str:
    return f"<think>\n{think}\n</think>\n\n<answer>\n{answer}\n</answer>"


GOOD_THINK = ("先比较本月销售额和小规模纳税人的免税线：本月9万元，没有超过10万元，所以结论是可以免征；"
              "这一步没有引入新的税率、金额、日期或相反结论。")


def test_lexicographic_order_holds():
    """规则通过 > 规则失败 > 答案漂移 > 格式失败，四档严格递减。"""
    good = wrap(GOOD_THINK, POOL[0])
    rule_bad = wrap(GOOD_THINK + " 这里故意插入 <img src=x> 这类图床痕迹。", POOL[0])
    answer_bad = wrap(GOOD_THINK, "本月销售额9万元，应缴增值税，税率3%。")
    format_bad = "<think>这段输出一直没有闭合，也没有答案"

    scores = [reward_one(text, USER_PROMPT, POOL, use_judge=False)
              for text in (good, rule_bad, answer_bad, format_bad)]
    assert scores[0] > scores[1] > scores[2] > scores[3]
    # 答案漂移必须是负分：正分意味着"漂了但写得好"仍可能被组内优势推上去
    assert scores[2] < 0


def test_missing_answer_tag_counts_as_format_failure():
    """只有 think、没有 answer 标签，和完全没闭合是同一档。

    这条防的是容错解析：宽松的解析器会把 </think> 之后那段裸文本当成 answer，
    于是一条根本没按格式写的输出照样能过答案门。
    """
    missing_tag = ("<think>" + GOOD_THINK + "</think>\n" + POOL[0])
    format_bad = "<think>这段输出一直没有闭合，也没有答案"
    assert (reward_one(missing_tag, USER_PROMPT, POOL, use_judge=False)
            == reward_one(format_bad, USER_PROMPT, POOL, use_judge=False))


def test_factless_answer_is_rejected_online():
    """没有可比事实的空话答案，在线要判负分。

    离线评测为了历史可比会放它过，在线不能——放过等于告诉模型
    "少说具体数字就能安全拿分"，正好把 grounding 反向优化掉。
    """
    vague = wrap(GOOD_THINK, "建议结合实际情况处理，具体请咨询主管税务机关。")
    assert reward_one(vague, USER_PROMPT, POOL, use_judge=False) < 0


def test_repeat_penalty_lowers_a_looping_think():
    """复读的 think 要比正常的低分。防的是撞到最大长度之后的退化。"""
    normal_score = reward_one(wrap(GOOD_THINK, POOL[0]), USER_PROMPT, POOL, use_judge=False)
    looping_score = reward_one(wrap("本月9万元没有超过10万元所以免征。" * 12, POOL[0]),
                               USER_PROMPT, POOL, use_judge=False)
    assert looping_score < normal_score
    assert normal_score > 0


def test_prompt_columns_expand_by_group_size():
    """数据列长度是 prompt 数，completions 长度是 prompt 数 × K，必须按整除展开。

    不展开就会错配：某条候选拿别的题的答案池去比，奖励算得完全是错的，
    而训练照常跑、曲线还挺好看。
    """
    rewards = RuleWarmupReward()(
        [wrap(GOOD_THINK, POOL[0]), "<think>坏格式"],
        user_prompt=[USER_PROMPT],
        v1_answers_json=[json.dumps(POOL, ensure_ascii=False)])
    assert len(rewards) == 2
    assert rewards[0] > rewards[1]
