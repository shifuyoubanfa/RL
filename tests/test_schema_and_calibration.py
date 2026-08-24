"""两组不变量：answer-lock 和标定判据。

它们都是"改坏了不报错、只是结果慢慢变歪"的那种东西，所以必须有用例钉住。
"""

from __future__ import annotations

from src.config import load_config
from src.data.parsing import parse_think_answer
from src.data.schema import dpo_row, grpo_row, sft_row
from src.rewards.calibration import (calibration_tiers, confident_cleaner, eval_se,
                                     sigma_of_score, true_gain_threshold)


class TestAnswerLock:
    def test_dpo_pair_shares_the_exact_same_answer(self):
        """一对偏好数据里，chosen 和 rejected 的答案段必须逐字相同。

        这是整条链路的核心不变量。答案段一旦跟着变，梯度就不只压 think 了，
        模型会学到"整段输出都可以改"，几百步之后答案开始漂。
        """
        answer = "本月销售额9万元，未超过10万元，免征增值税。"
        row = dpo_row("题面", "干净的推理", "带检索腔的推理", answer)
        chosen_answer = parse_think_answer(row["messages"][-1]["content"])[1]
        rejected_answer = parse_think_answer(row["rejected_response"])[1]
        assert chosen_answer == rejected_answer == answer

    def test_dpo_pair_differs_only_in_think(self):
        row = dpo_row("题面", "干净的推理", "带检索腔的推理", "答案")
        assert parse_think_answer(row["messages"][-1]["content"])[0] == "干净的推理"
        assert parse_think_answer(row["rejected_response"])[0] == "带检索腔的推理"

    def test_sft_row_keeps_the_opening_think_tag(self):
        """开头那个 <think> 必须在训练目标里。

        少了它，学出来的模型不吐开标签，下游所有解析全部失效——
        而且这种失效表现为"评测分数莫名其妙全崩"，不容易一眼看出根因。
        """
        content = sft_row("题面", "推理", "答案")["messages"][-1]["content"]
        assert content.startswith("<think>")
        assert "</think>" in content and "<answer>" in content

    def test_grpo_row_has_no_assistant_turn(self):
        """GRPO 行不带答案——答案是训练时现场采的。"""
        row = grpo_row("q1", "问题", "题面", "金标准", ["池答案A", "池答案B"])
        assert [m["role"] for m in row["messages"]] == ["system", "user"]
        assert row["pool_size"] == 2


class TestCalibration:
    def test_tiers_are_monotonically_cleaner(self):
        means = [mean for mean, _sigma in calibration_tiers()]
        assert means == sorted(means), "标定表必须由脏到干净排列，查表插值依赖这个顺序"

    def test_sigma_grows_towards_the_clean_end(self):
        """越靠近"完全没照抄"这一端，裁判的方差越大。

        这条不是巧合，是这套判据存在的理由：干净端两条质量相仿的 think 能差一两分，
        不查表直接比大小就会选进一批"只是这次抖高了"的样本。
        """
        assert sigma_of_score(7.0) > sigma_of_score(1.0)

    def test_score_outside_the_table_is_clamped_not_extrapolated(self):
        tiers = calibration_tiers()
        assert sigma_of_score(-5.0) == tiers[0][1]
        assert sigma_of_score(99.0) == tiers[-1][1]

    def test_confident_cleaner_rejects_a_small_gap(self):
        # 干净端 σ 大，差 0.3 分远远不够两条误差带错开
        assert confident_cleaner(7.3, 7.0, 2.0) is False

    def test_confident_cleaner_accepts_a_large_gap(self):
        # 干净端对上"抄了四句"那一档，差距远超噪声带
        assert confident_cleaner(7.3, 0.03, 2.0) is True

    def test_confident_cleaner_rejects_when_candidate_is_worse(self):
        assert confident_cleaner(2.0, 7.0, 2.0) is False

    def test_more_repeats_barely_help_the_overall_standard_error(self):
        """把 k 从 3 加到 16，标准误几乎不动。

        瓶颈是题与题之间的差异，k 压不掉它。这条用例把"加 k 没用、要加题数"这个结论钉住，
        免得有人为了让判据变严去把评测的 k 调大——那是纯烧钱。
        """
        se_k3 = eval_se(500, 3)
        se_k16 = eval_se(500, 16)
        assert (se_k3 - se_k16) / se_k3 < 0.10

    def test_more_items_do_help(self):
        assert eval_se(2000, 3) < eval_se(500, 3)

    def test_true_gain_threshold_is_three_standard_errors(self):
        n, k = 500, 3
        assert abs(true_gain_threshold(n, k) - 3 * eval_se(n, k)) < 1e-9


class TestConfig:
    def test_placeholders_are_expanded(self):
        cfg = load_config(reload=True)
        assert "${" not in str(cfg.paths["output_dir"])
        assert str(cfg.paths["output_dir"]).startswith(str(cfg.paths["work_dir"]))

    def test_dotted_lookup(self):
        cfg = load_config(reload=True)
        assert cfg.get_path("train.sft.epochs") == cfg.train["sft"]["epochs"]
        assert cfg.get_path("train.nope.nope", "fallback") == "fallback"

    def test_gate_k_values_respect_the_calibration_requirement(self):
        """选样 k 必须 ≥ 16，否则 σ 标定表不作数。配置里改小了要在这里就炸。"""
        cfg = load_config(reload=True)
        assert int(cfg.gates["k_select"]) >= 16
        assert int(cfg.gates["k_screen"]) < int(cfg.gates["k_select"])
