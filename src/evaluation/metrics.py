"""评测第二步：三件套判分，产人读报告和机读摘要。

本模块在整条链路里的位置：评测的后半段，也是整条链路唯一的成绩单。

**三件套，一个工具只判一件事：**

| 指标 | 谁判的 | 判什么 | 有没有噪声 |
|---|---|---|---|
| 干净分 | 裁判模型，每题打 `k_eval` 遍 | 换词复述照抄的程度，0~10 | 有，标准误见下 |
| 规则通过率 | 确定性规则 | think 里有没有检索腔表面标记 | 无 |
| 答案在池率 | 确定性规则 | 答案的极性和数字还在不在基座认可池里 | 无 |

**为什么必须三个一起看，不能只留一个。** 只看干净分，模型可以把 think 写成一段和参考
毫无关系的空话——照抄为零，分很高，但答案早飘了。只看规则通过率，模型只要不说
"参考问答对"四个字就满分，换词复述照样满篇。只看在池率，模型原地不动就是满分。
三个指标各自都能被单独刷高，但要同时刷高，只有真把 think 写好这一条路。

**真涨门。** 干净分是有噪声的，两个阶段的均值差要大过约 3 倍标准误才算真涨，
否则记统计打平，不许说谁更好。门限由 :func:`src.rewards.calibration.true_gain_threshold`
按题数和 k 算出来，写在报告里。另外两个指标是确定性的，没有打分噪声，直接比数就行。

**格式失败怎么记账。** 生成被截断在 think 里的输出：干净分照常打（评的是它真写出来的那段
残缺推理），规则强制判失败，答案判不在池。报告里单列格式完整率——不这么记，
一个格式全崩的模型会因为"残缺 think 里没有检索腔词"而拿到很高的规则通过率。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.config import Config
from src.data.build_answer_pool import load_pool_index
from src.data.jsonl_io import gather_until, qid_of, read_jsonl, write_jsonl
from src.data.parsing import extract_references
from src.rewards.calibration import eval_se, true_gain_threshold
from src.rewards.judge import score_think
from src.rewards.rules import answer_in_v1_pool, detect_rag_style
from src.utils.log import get_logger

log = get_logger("eval_metrics")

# 报告里必须出现的三个标记。编排器靠它们判断"这份报告写全了没有"，
# 用来决定续跑时要不要重评。改了报告措辞就要同步改这里。
REPORT_MARKERS = ("干净分", "规则去检索腔通过率", "答案在池率")


def score_record(cfg: Config, record: dict, pool_index: dict[str, dict]) -> dict:
    """给一条推理结果打三件套的分。

    :param cfg: 配置
    :param record: 一条推理输出，含 think/answer/format_ok
    :param pool_index: qid 到答案池的映射
    :return: 逐条判分结果
    """
    think = record.get("think") or ""
    answer = record.get("answer") or ""
    references = extract_references(record.get("user_prompt") or "")
    qid = record.get("qid") or qid_of(record.get("query") or "")
    pool_answers = (pool_index.get(qid) or {}).get("pool_answers") or []

    judged = score_think(references, think, int(cfg.gates["k_eval"]))
    rule = detect_rag_style(think)

    format_ok = bool(record.get("format_ok", bool(think.strip() and answer.strip())))
    format_reason = str(record.get("format_reason") or ("ok" if format_ok else "unmarked_empty"))
    # 格式失败或 think 为空 → 规则这一项强制判失败。见模块开头"格式失败怎么记账"。
    rule_forced_failure = (not format_ok) or (not think.strip())
    rule_pass = (not rule_forced_failure) and (not rule["has_rag_style"])

    if pool_answers:
        verdict = answer_in_v1_pool(answer, pool_answers)
        in_pool, comparable = verdict["in_pool"], verdict["comparable"]
        drift, answer_reason, no_pool = verdict["drift_facts"], verdict["reason"], False
    else:
        # 缺池的题单独标出来，不进在池率的分母。混进去等于把"测不了"算成"没通过"。
        in_pool, comparable, drift, answer_reason, no_pool = None, False, [], "no_pool", True

    return {
        "qid": qid, "query": record.get("query"),
        "clean_score": judged["clean_score"], "clean_n": judged["n"],
        "has_rag_style": rule["has_rag_style"], "n_traces": rule["n"],
        "rule_pass": rule_pass, "rule_forced_failure": rule_forced_failure,
        "format_ok": format_ok, "format_reason": format_reason,
        "empty_think": not bool(think.strip()), "empty_answer": not bool(answer.strip()),
        "in_pool": in_pool, "answer_comparable": comparable, "answer_reason": answer_reason,
        "drift_facts": drift, "no_pool": no_pool,
    }


def _aggregate(cfg: Config, scored: list[dict], tag: str) -> dict:
    """把逐条判分聚合成摘要。

    干净分先按 `clean_n > 0` 过滤再求均值。这一步不能省：判分全失败时函数返回的是 None，
    而 None 若被当成 0 混进去，恰好落在标定表"整段照抄"那一档，一次服务故障就能把
    整体均值拉低半分。

    :param cfg: 配置
    :param scored: 逐条判分结果
    :param tag: 模型标签
    :return: 机读摘要
    """
    n = len(scored)
    valid = [r["clean_score"] for r in scored if r["clean_n"] > 0 and r["clean_score"] is not None]

    rule_pass = sum(1 for r in scored if r["rule_pass"])
    n_pool = sum(1 for r in scored if not r["no_pool"])
    in_pool = sum(1 for r in scored if not r["no_pool"] and r["in_pool"])
    n_comparable = sum(1 for r in scored if not r["no_pool"] and r.get("answer_comparable", True))
    in_pool_comparable = sum(1 for r in scored if not r["no_pool"]
                             and r.get("answer_comparable", True) and r["in_pool"])
    n_format_failure = sum(1 for r in scored if not r.get("format_ok"))
    k_eval = int(cfg.gates["k_eval"])

    return {
        "tag": tag, "n": n, "n_valid_judge": len(valid),
        "n_pool": n_pool, "n_no_pool": n - n_pool,
        "n_answer_comparable": n_comparable, "n_answer_uncomparable": n_pool - n_comparable,
        "n_empty_answer": sum(1 for r in scored if r.get("answer_reason") == "empty_answer"),
        "n_format_failure": n_format_failure,
        "format_pass_rate": round((n - n_format_failure) / max(1, n), 4),
        "format_failure_reasons": dict(Counter(
            r.get("format_reason") or "unknown" for r in scored if not r.get("format_ok"))),
        "clean_mean": round(sum(valid) / len(valid), 4) if valid else None,
        "rule_pass_rate": round(rule_pass / max(1, n), 4),
        "in_pool_rate": round(in_pool / max(1, n_pool), 4),
        "in_pool_comparable_rate": round(in_pool_comparable / max(1, n_comparable), 4),
        "se": round(eval_se(n, k_eval), 4),
        "true_gain_threshold": round(true_gain_threshold(n, k_eval), 4),
        "k_eval": k_eval,
    }


def _render_report(summary: dict) -> str:
    """把摘要渲染成人读的 Markdown 报告。

    每个数字都带上"谁评的 + 样本量"。少了这两样，一个数字在会上就说不清是什么意思。

    :param summary: 机读摘要
    :return: Markdown 文本
    """
    n = summary["n"]
    n_pool = summary["n_pool"]
    lines = [
        f"# 三件套评测报告 · {summary['tag']}",
        "",
        f"- 样本数 N = {n}（裁判有效打分 {summary['n_valid_judge']} 条；"
        f"缺答案池 {summary['n_no_pool']} 题；空答案 {summary['n_empty_answer']} 题）",
        f"- **生成格式完整率** = {n - summary['n_format_failure']}/{n} = "
        f"{summary['format_pass_rate']:.1%}"
        f"（格式失败 {summary['n_format_failure']} 题：裁判照评残缺 think，规则强制判失败，空答案判不在池）",
        f"- **裁判干净分（k={summary['k_eval']}，N={n}）** = "
        f"{summary['clean_mean'] if summary['clean_mean'] is not None else 'N/A'}"
        f"（0-10，越高越没有换词复述照抄；标准误 ≈ {summary['se']:.3f}，"
        f"两阶段差值大于 {summary['true_gain_threshold']:.2f} 才算真涨）",
        f"- **规则去检索腔通过率（规则，N={n}）** = "
        f"{round(summary['rule_pass_rate'] * n)}/{n} = {summary['rule_pass_rate']:.1%}",
        f"- **答案在池率（规则，N={n_pool}）** = "
        f"{round(summary['in_pool_rate'] * n_pool)}/{n_pool} = {summary['in_pool_rate']:.1%}"
        f"（漂移率 {1 - summary['in_pool_rate']:.1%}）",
        f"- 答案可比较审计 = {summary['n_answer_comparable']}/{n_pool}；"
        f"可比较题的在池率 = {summary['in_pool_comparable_rate']:.1%}"
        f"（没有极性/数字/日期的非空回答不改主指标，只单列披露）",
        "",
        "口径：参考资料取自题面【参考问答对】段；干净分复用与标定同一套判分提示词；"
        "训练样本的答案段锁死基座原版，只训 think。",
    ]
    return "\n".join(lines) + "\n"


def run(cfg: Config, *, infer_path: Path, scores_path: Path, report_path: Path,
        summary_path: Path, tag: str) -> dict:
    """跑三件套判分。

    :param cfg: 配置
    :param infer_path: 推理结果
    :param scores_path: 逐条判分输出
    :param report_path: 人读报告输出
    :param summary_path: 机读摘要输出
    :param tag: 模型标签
    :return: 机读摘要
    :raise SystemExit: 推理结果为空，或裁判判分全部失败
    """
    records = read_jsonl(infer_path)
    for record in records:                             # 补 qid，进度文件按它续跑
        record.setdefault("qid", qid_of(record.get("query") or ""))
    pool_index = load_pool_index(cfg)
    log.info("三件套评测：%d 条（tag=%s）", len(records), tag)

    # 评测这一档的判分不进缓存（k 小、一次性），靠进度文件让中断不重烧
    scored = gather_until(
        records, lambda r: score_record(cfg, r, pool_index),
        enough=lambda _rs: False,                      # 恒 False = 全评，不早停
        chunk=int(cfg.data["gather_chunk"]),
        workers=int(cfg.judge["call_workers"]),
        desc=f"eval:{tag}",
        progress_path=Path(scores_path).with_name(f"{Path(scores_path).stem}_progress.jsonl"))
    write_jsonl(scores_path, scored)

    summary = _aggregate(cfg, scored, tag)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    if summary["n"] == 0 or summary["clean_mean"] is None:
        # 写一份不含 REPORT_MARKERS 的失败报告：编排器据此判"没评成"，续跑会重评。
        # 绝不把"外部服务全挂"伪装成"已评测、分数是 nan"。
        Path(report_path).write_text(
            f"# 三件套评测 FAILED · {tag}\n"
            f"N={summary['n']}，裁判有效打分={summary['n_valid_judge']}。"
            f"推理结果为空或判分全部失败，未产出有效分数。查推理服务和裁判服务后重评。\n",
            encoding="utf-8")
        log.error("评测无有效分（N=%d valid=%d），已写失败报告：%s",
                  summary["n"], summary["n_valid_judge"], report_path)
        raise SystemExit(f"eval {tag}: 没有有效的裁判打分")

    # 摘要先落盘，报告后落盘。顺序反了的话，编排器看到报告完整就去读摘要，可能读到不存在的文件。
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(report_path).write_text(_render_report(summary), encoding="utf-8")

    log.info("干净分=%.3f | 规则通过=%.1f%% | 在池率=%.1f%%（分母 %d）-> %s",
             summary["clean_mean"], 100 * summary["rule_pass_rate"],
             100 * summary["in_pool_rate"], summary["n_pool"], report_path)
    return summary


def report_is_complete(report_path: Path, summary_path: Path) -> bool:
    """这份评测是不是已经跑完了。

    两个条件都要满足：报告里三个标记齐全，摘要文件存在。只看报告不够——
    摘要是阶段门要读的，缺了它编排器会拿不到数。

    :param report_path: 人读报告
    :param summary_path: 机读摘要
    :return: 完整为 True
    """
    if not Path(report_path).exists() or not Path(summary_path).exists():
        return False
    text = Path(report_path).read_text(encoding="utf-8", errors="replace")
    return all(marker in text for marker in REPORT_MARKERS)
