"""在线 GRPO 数据组装的契约测试：三条拒绝必须是硬失败。

数据泄漏和缺池这两件事的共同点是：出了不报错，只是让后面所有数字失去意义。
所以这一层选择直接退出，而不是打个警告继续跑。
"""

from __future__ import annotations

import json

import pytest

from src.data.build_grpo_data import build_rows, pool_is_trainable

CONCRETE_POOL = ["本月销售额9万元，未超过10万元，可以免征增值税。"]
VAGUE_POOL = ["需结合实际情况判断。", "请按政策规定处理。"]


def test_pool_with_concrete_facts_is_trainable():
    assert pool_is_trainable(CONCRETE_POOL) is True


def test_vague_pool_is_not_trainable():
    """池里抽不出任何极性、数字、日期，硬门就没有靶子。

    留着这类题，模型会学到"少说具体数字就能安全拿分"，恰好把 grounding 反向优化掉。
    """
    assert pool_is_trainable(VAGUE_POOL) is False


def test_pool_merges_canonical_gold_and_samples_in_order():
    rows = build_rows(
        [{"qid": "q1", "query": "问题", "user_prompt": "题面", "split": "train",
          "gold_answer": "本月销售额9万元，未超过10万元，可以免征增值税。"}],
        {"q1": {
            "canonical_answer": "本月销售额9万元，未超过10万元，可以免征增值税。",
            "gold_answer": "销售额9万元低于10万元标准，因此免征增值税。",
            "pool_answers": ["本月9万元没有超过10万元，可以免税。",
                             "本月9万元没有超过10万元，可以免税。", ""],
        }},
        shuffle_seed=None)

    pool = json.loads(rows[0]["v1_answers_json"])
    # 贪心答案排第一、重复的和空的去掉
    assert pool == [
        "本月销售额9万元，未超过10万元，可以免征增值税。",
        "销售额9万元低于10万元标准，因此免征增值税。",
        "本月9万元没有超过10万元，可以免税。",
    ]


def test_untrainable_rows_are_skipped_not_fatal():
    rows = build_rows(
        [{"qid": "q1", "query": "a", "user_prompt": "u1", "split": "train", "gold_answer": "需结合实际判断。"},
         {"qid": "q2", "query": "b", "user_prompt": "u2", "split": "train", "gold_answer": CONCRETE_POOL[0]}],
        {"q1": {"pool_answers": VAGUE_POOL}, "q2": {"pool_answers": CONCRETE_POOL}},
        shuffle_seed=None)
    assert [r["qid"] for r in rows] == ["q2"]


def test_eval_qid_overlap_is_fatal():
    """验收题混进训练数据必须直接退出。

    在线训练会把同一道题反复采几十次。一旦泄漏，后面所有评测数字都不作数，
    而且从曲线上完全看不出来。
    """
    with pytest.raises(SystemExit):
        build_rows(
            [{"qid": "leaked", "query": "a", "user_prompt": "u", "split": "train", "gold_answer": "g"}],
            {"leaked": {"pool_answers": CONCRETE_POOL}},
            eval_qids={"leaked"}, shuffle_seed=None)


def test_missing_pool_is_fatal():
    with pytest.raises(SystemExit):
        build_rows(
            [{"qid": "q1", "query": "a", "user_prompt": "u", "split": "train", "gold_answer": "g"}],
            {}, shuffle_seed=None)


def test_all_filtered_out_is_fatal():
    with pytest.raises(SystemExit):
        build_rows(
            [{"qid": "q1", "query": "a", "user_prompt": "u", "split": "train", "gold_answer": "g"}],
            {"q1": {"pool_answers": VAGUE_POOL}}, shuffle_seed=None)
