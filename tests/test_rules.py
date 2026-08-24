"""确定性规则层的契约测试。

这些用例全是**离线用例**：手写输入、手写期望，不连任何服务，不代表模型真实表现。
它们保的是"规则的判定口径不许被无意改掉"——这一层同时是训练硬门和评测指标，
口径一漂，训练信号和成绩单会一起错，而且两边错得一致，看不出来。
"""

from __future__ import annotations

import pytest

from src.rewards.rules import answer_in_v1_pool, detect_rag_style, extract_facts

POOL = [
    "本月销售额9万元，未超过10万元，免征增值税。",
    "销售额为9万元，可以享受免税政策。",
    "本月9万元，无需缴纳增值税。",
]


class TestDetectRagStyle:
    def test_clean_think_has_no_trace(self):
        think = "先看这个月卖了多少——9万；小规模这档免税线是10万，9万还在线里，那这部分增值税就免了。"
        assert detect_rag_style(think)["has_rag_style"] is False

    def test_explicit_retrieval_phrase_is_caught(self):
        think = "根据参考问答对1，资料显示月销售额未超过10万元的免征增值税。"
        result = detect_rag_style(think)
        assert result["has_rag_style"] is True
        assert "A_检索装置腔" in result["n_by_type"]

    def test_single_citation_inside_reasoning_is_not_a_trace(self):
        # 嵌进推理里的合法引用不能算痕迹。把它算成痕迹，模型会学到"别引用法条"，
        # grounding 跟着塌掉——这是这一层最重要的一条豁免。
        think = "按《增值税暂行条例》的规定，这一档适用3%，所以本月应当按3%计算。"
        assert detect_rag_style(think)["has_rag_style"] is False

    def test_policy_label_list_is_a_trace(self):
        think = "结论如上。\n政策依据：财税〔2023〕1号"
        result = detect_rag_style(think)
        assert result["has_rag_style"] is True
        assert "D_清单式甩文号" in result["n_by_type"]

    def test_image_link_is_case_insensitive(self):
        assert detect_rag_style("详见 TABLE1.PNG 的数据")["has_rag_style"] is True


class TestExtractFacts:
    def test_long_polarity_word_blocks_its_substring(self):
        # "不超过" 不能同时抽出 "超过"，否则一句话会带上互相矛盾的两个极性
        facts = extract_facts("本月销售额不超过10万元")
        assert "不超过" in facts["polarity"]
        assert "超过" not in facts["polarity"]

    def test_money_unit_is_normalized(self):
        assert extract_facts("10万元")["value"] == extract_facts("10万")["value"]

    def test_date_is_consumed_before_duration(self):
        # 日期先抽并消位，否则期限正则会从 "2023年1月1日" 里抽出根本不存在的 "1月"
        facts = extract_facts("自2023年1月1日起执行")
        assert "2023年1月1日" in facts["date"]
        assert not any(v.endswith("月") for v in facts["value"])


class TestAnswerInPool:
    def test_matching_answer_passes(self):
        result = answer_in_v1_pool("本月销售额9万元，免征增值税。", POOL)
        assert result["in_pool"] is True
        assert result["comparable"] is True

    def test_empty_answer_fails(self):
        # 空集合是任何集合的子集。不特判，空答案会白拿一个通过。
        result = answer_in_v1_pool("", POOL)
        assert result["in_pool"] is False
        assert result["reason"] == "empty_answer"

    def test_new_number_is_drift(self):
        result = answer_in_v1_pool("税率为13%，应缴增值税。", POOL)
        assert result["in_pool"] is False
        assert "13%" in result["drift_facts"]

    def test_factless_answer_keeps_metric_but_is_flagged_uncomparable(self):
        # 主指标沿用旧口径以保持历史可比，但必须单列出来审计，
        # 不能把"规则测不到"伪装成"通过"。
        result = answer_in_v1_pool("建议结合实际情况处理。", POOL)
        assert result["in_pool"] is True
        assert result["comparable"] is False
        assert result["reason"] == "no_comparable_facts"

    def test_date_granularity_is_not_drift(self):
        result = answer_in_v1_pool("自2023年起免征。", ["自2023年1月1日起免征增值税。"])
        assert result["in_pool"] is True

    @pytest.mark.parametrize("answer", ["请咨询主管税务机关。", "需要结合具体情况判断。"])
    def test_no_facts_on_either_side_is_uncomparable(self, answer):
        result = answer_in_v1_pool(answer, ["请结合实际情况判断。"])
        assert result["comparable"] is False
