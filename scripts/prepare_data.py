#!/usr/bin/env python
"""数据处理的唯一入口。

用法：

    python scripts/prepare_data.py <步骤> [--config configs/train.yaml] [--limit N]

七个步骤，按顺序：

    base-outputs   基座对全部题重产 think/answer            需要推理服务
    split          切训练集 / 冻结验收集 / 建池题集          纯 CPU
    answer-pool    基座对每题采多条，建认可答案池            需要推理服务
    sft-data       裁判改写 + 三道门 → 冷启动 SFT 数据       需要裁判服务
    rft-data       从自采样结果里筛 → 拒绝采样 SFT 数据      需要裁判服务
    dpo-pairs      从 rollout 里筛 → 偏好对                  需要裁判服务
    grpo-data      训练题集 + 答案池 → 在线 GRPO prompt      纯 CPU

`all` 会按顺序跑完这七步，但**不含两次 rollout 采样**——那两次采样要用刚训完的模型，
必须夹在训练中间，所以由 `scripts/train.sh` 统一编排。单独跑数据步一般是为了排查某一步，
真要端到端跑就直接用 `train.sh`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dirs, load_config          # noqa: E402
from src.data import (build_answer_pool, build_base_outputs, build_dpo_pairs,  # noqa: E402
                      build_grpo_data, build_rft_data, build_sft_data, split_dataset)

# 步骤名 → (处理函数, 支不支持 --limit)
_STEPS = {
    "base-outputs": (build_base_outputs.run, True),
    "split": (split_dataset.run, False),
    "answer-pool": (build_answer_pool.run, False),
    "sft-data": (build_sft_data.run, True),
    "rft-data": (build_rft_data.run, False),
    "dpo-pairs": (build_dpo_pairs.run, False),
    "grpo-data": (build_grpo_data.run, True),
}
# `all` 的执行顺序。字典本身是有序的，但把顺序单独写出来更不容易被无意改动。
_ALL_ORDER = ("base-outputs", "split", "answer-pool", "sft-data", "rft-data", "dpo-pairs", "grpo-data")


def main() -> int:
    parser = argparse.ArgumentParser(description="数据处理入口", epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=[*_STEPS, "all"], help="要跑哪一步")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 configs/train.yaml")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 条，0 = 全量。联调用，部分步骤支持")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    steps = _ALL_ORDER if args.step == "all" else (args.step,)
    for name in steps:
        handler, supports_limit = _STEPS[name]
        print(f"\n===== {name} =====", flush=True)
        if supports_limit and args.limit:
            handler(cfg, limit=args.limit)
        else:
            handler(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
