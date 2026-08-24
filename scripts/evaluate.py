#!/usr/bin/env python
"""评测的唯一入口：在冻结验收集上跑一个模型，出三件套成绩单。

用法：

    # 已经 serve 好模型，直接评
    python scripts/evaluate.py --tag dpo --served-name dpo

    # 只判分，不重新推理（推理结果已有）
    python scripts/evaluate.py --tag dpo --skip-infer

    # 比较两个阶段，看差值过没过真涨门
    python scripts/evaluate.py --compare rft dpo

三件套（每个指标都标了谁评的）：

    裁判干净分     裁判模型，每题打 k 遍   换词复述照抄的程度，0~10，有噪声
    规则通过率     确定性规则             think 里有没有检索腔表面标记，无噪声
    答案在池率     确定性规则             答案还在不在基座认可池里，无噪声

推理需要一个已经 serve 起来的推理服务（`scripts/serve_vllm.sh`），判分只需要裁判服务。
`--served-name` 决定套哪套 system 提示，见 src/data/prompts.py 的 system_for——
评基座要传配置里的 `vllm.served_model`，评训练后的模型传别的任意名字。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dirs, load_config        # noqa: E402
from src.data import paths                             # noqa: E402
from src.evaluation import infer as eval_infer         # noqa: E402
from src.evaluation import metrics as eval_metrics     # noqa: E402
from src.rewards.calibration import true_gain_threshold  # noqa: E402


def _compare(cfg, tag_before: str, tag_after: str) -> int:
    """比较两个阶段的成绩单，按真涨门判读。

    干净分是有噪声的，差值要过 3 倍标准误才敢说涨了；另外两个指标是确定性的，
    直接比数即可，所以下面对它们不套统计判据。

    :return: 进程退出码，缺摘要返回 1
    """
    _, _, _, summary_before = paths.eval_paths(cfg, tag_before)
    _, _, _, summary_after = paths.eval_paths(cfg, tag_after)
    for path in (summary_before, summary_after):
        if not path.exists():
            print(f"缺评测摘要: {path}", file=sys.stderr)
            return 1

    before = json.loads(summary_before.read_text(encoding="utf-8"))
    after = json.loads(summary_after.read_text(encoding="utf-8"))
    threshold = true_gain_threshold(after["n"], after["k_eval"])
    delta = after["clean_mean"] - before["clean_mean"]
    verdict = "真涨" if delta > threshold else ("真降" if delta < -threshold else "统计打平")

    print(f"\n{tag_before} → {tag_after}（N={after['n']}, k={after['k_eval']}）")
    print(f"  裁判干净分   {before['clean_mean']:.3f} → {after['clean_mean']:.3f}"
          f"  差 {delta:+.3f}  真涨门 {threshold:.3f}  → {verdict}")
    print(f"  规则通过率   {before['rule_pass_rate']:.1%} → {after['rule_pass_rate']:.1%}"
          f"  差 {after['rule_pass_rate'] - before['rule_pass_rate']:+.1%}（确定性，无噪声）")
    print(f"  答案在池率   {before['in_pool_rate']:.1%} → {after['in_pool_rate']:.1%}"
          f"  差 {after['in_pool_rate'] - before['in_pool_rate']:+.1%}（确定性，无噪声）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="评测入口", epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--tag", help="这次评测的标签，决定产物文件名")
    parser.add_argument("--served-name", default=None,
                        help="模型在推理服务里的名字，决定套哪套 system 提示；默认与 --tag 相同")
    parser.add_argument("--eval-file", default=None, help="覆盖冻结验收集路径")
    parser.add_argument("--skip-infer", action="store_true", help="推理结果已有，只判分")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="比较两个已评标签的成绩单")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if args.compare:
        return _compare(cfg, *args.compare)
    if not args.tag:
        parser.error("需要 --tag（或用 --compare）")

    infer_path, scores_path, report_path, summary_path = paths.eval_paths(cfg, args.tag)
    if not args.skip_infer:
        eval_infer.run(cfg, model_name=args.served_name or args.tag,
                       eval_file=Path(args.eval_file) if args.eval_file else paths.eval_set(cfg),
                       out_path=infer_path)
    summary = eval_metrics.run(cfg, infer_path=infer_path, scores_path=scores_path,
                               report_path=report_path, summary_path=summary_path, tag=args.tag)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n报告 -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
