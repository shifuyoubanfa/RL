"""评测记账的契约测试：格式失败不许被算成通过。

这一组保的是"指标不许虚高"。一条撞到最大生成长度、卡在 think 里就断了的输出，
如果被容错解析当成"有答案"，它会同时拿到规则通过和答案在池两个通过——
一个格式全崩的模型，成绩单上反而好看。
"""

from __future__ import annotations

from unittest.mock import patch

from src.config import load_config
from src.data.jsonl_io import qid_of
from src.data.parsing import parse_think_answer, parse_think_answer_diagnostic
from src.evaluation import metrics


class TestDiagnosticParsing:
    def test_truncated_think_never_masquerades_as_answer(self):
        raw = "<think>部分推理反复循环，直到达到最大生成长度"
        parsed = parse_think_answer_diagnostic(raw)
        assert parsed["think"] == "部分推理反复循环，直到达到最大生成长度"
        assert parsed["answer"] == ""
        assert parsed["format_ok"] is False
        assert parsed["format_reason"] == "missing_think_close+empty_answer"

    def test_tolerant_parser_keeps_its_documented_behaviour(self):
        # 容错版对同一条输入会把整段（连开标签一起）当 answer 交出去。
        # 这是它的既定行为，数据构建阶段用得上；这条用例把两者的差别钉死，
        # 防止有人把评测侧换成容错版——那样一条彻底没写完的输出会被算成"有答案"。
        think, answer = parse_think_answer("<think>没写完的推理")
        assert think == ""
        assert answer == "<think>没写完的推理"

    def test_wellformed_output_parses_cleanly(self):
        parsed = parse_think_answer_diagnostic("<think>\n推理\n</think>\n\n<answer>\n答复\n</answer>")
        assert parsed == {"think": "推理", "answer": "答复", "format_ok": True, "format_reason": "ok"}


class TestScoreRecord:
    def test_format_failure_forces_rule_and_answer_failure(self):
        """格式失败时：裁判分照记（它评的是真写出来的那段），规则和答案强制判失败。"""
        cfg = load_config()
        query = "测试题"
        record = {
            "query": query, "qid": qid_of(query),
            "user_prompt": "【参考问答对】参考【问题】测试题",
            "think": "一段没有任何规则关键词的残缺推理",
            "answer": "",
            "format_ok": False,
            "format_reason": "missing_think_close+empty_answer",
        }
        pool_index = {qid_of(query): {"pool_answers": ["应当缴纳3%的税款"]}}
        with patch.object(metrics, "score_think", return_value={"clean_score": 4.0, "n": 3}):
            scored = metrics.score_record(cfg, record, pool_index)

        assert scored["clean_score"] == 4.0
        assert scored["has_rag_style"] is False        # 残缺 think 里确实没有检索腔词
        assert scored["rule_pass"] is False            # 但仍然强制判失败
        assert scored["rule_forced_failure"] is True
        assert scored["in_pool"] is False
        assert scored["answer_reason"] == "empty_answer"

    def test_missing_pool_is_excluded_not_failed(self):
        """缺答案池的题要退出分母，不能算成"没通过"。"""
        cfg = load_config()
        record = {"query": "无池题", "qid": qid_of("无池题"), "user_prompt": "题面",
                  "think": "一段推理", "answer": "一个答复", "format_ok": True, "format_reason": "ok"}
        with patch.object(metrics, "score_think", return_value={"clean_score": 6.0, "n": 3}):
            scored = metrics.score_record(cfg, record, {})
        assert scored["no_pool"] is True
        assert scored["in_pool"] is None
        assert scored["answer_reason"] == "no_pool"


class TestAggregate:
    def test_failed_judge_calls_do_not_drag_the_mean_to_zero(self):
        """判分全失败的样本返回 None，聚合时必须先过滤。

        不过滤而当成 0 混进去，恰好落在标定表"整段照抄"那一档，
        一次服务故障就能把整体均值拉低半分。
        """
        cfg = load_config()
        scored = [
            {"clean_score": 6.0, "clean_n": 3, "rule_pass": True, "rule_forced_failure": False,
             "format_ok": True, "format_reason": "ok", "in_pool": True, "no_pool": False,
             "answer_comparable": True, "answer_reason": "ok"},
            {"clean_score": None, "clean_n": 0, "rule_pass": True, "rule_forced_failure": False,
             "format_ok": True, "format_reason": "ok", "in_pool": True, "no_pool": False,
             "answer_comparable": True, "answer_reason": "ok"},
        ]
        summary = metrics._aggregate(cfg, scored, "unit-test")
        assert summary["clean_mean"] == 6.0
        assert summary["n"] == 2
        assert summary["n_valid_judge"] == 1
