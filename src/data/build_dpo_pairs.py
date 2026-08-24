"""第 6 步：构 DPO 偏好对。

本模块在整条链路里的位置：四阶段的第三阶段，也是第一个真正的偏好学习阶段。

**一对偏好数据长这样：**

- ``chosen``  = (筛出来的干净 think, 基座原 answer)
- ``rejected``= (基座原 think,       基座原 answer)

两边共用同一段 answer，只有 think 不同。这就是 answer-lock（见 :mod:`src.data.schema`）。

**为什么 rejected 直接用基座原推理，而不是再采一批差样本。** 三个理由：

1. **零成本**。基座原推理在第 1 步就有了，不用额外生成、额外打分。
2. **天然分得开**。基座原推理就是满是检索腔的那一版，和洗干净的 chosen 之间的差距远超噪声带。
   如果 rejected 也是从当前模型采出来的，chosen 和 rejected 会挤在干净这一端，
   裁判分不开谁好谁坏，学出来的是噪声。
3. **对比维度是干净的**。一对里只有 think 不同，模型没有别的维度可以走捷径。

**这一步挡不住什么**：它教不了模型"更干净"是什么样子的**上限**。chosen 来自当前模型自己的
采样，模型写不出来的东西不会出现在 chosen 里。要往上顶，得靠在线阶段每步重新采样。

输入：`30_dpo_rollout.jsonl`（由 :mod:`src.data.rollout` 产出，采样自上一阶段的模型）。
输出：`31_dpo_pairs.jsonl`。
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.data import paths
from src.data.build_answer_pool import load_pool_index
from src.data.jsonl_io import gather_until, index_by, qid_of, read_jsonl, write_jsonl
from src.data.parsing import extract_references
from src.data.schema import dpo_row
from src.data.selection import select_clean_think
from src.utils.log import get_logger

log = get_logger("build_dpo_pairs")

# 偏好对少于这个数就告警：对数太少，DPO 那点梯度推不动一个 32B 模型。
_MIN_USABLE_PAIRS = 50


def run(cfg: Config) -> Path:
    """跑第 6 步。

    :param cfg: 配置
    :return: 偏好对文件路径
    """
    rollouts = read_jsonl(paths.dpo_rollout(cfg))
    for record in rollouts:                            # rollout 里可能只有 query，补上 qid 供续跑去重
        record.setdefault("qid", qid_of(record.get("query") or ""))
    train_index = index_by(read_jsonl(paths.train_set(cfg)))
    pool_index = load_pool_index(cfg)
    target = int(cfg.train["dpo"]["target_pairs"])

    log.info("构偏好对：%d 题，目标 %d 对", len(rollouts), target)

    def _pair(record: dict) -> dict:
        qid = record.get("qid")
        base = train_index.get(qid) or {}
        base_think = (base.get("reasoning") or "").strip()
        out = {"qid": qid, "query": record.get("query"),
               "user_prompt": record.get("user_prompt"),
               "base_think": base_think,
               "answer": (base.get("answer") or "").strip(),
               "chosen_think": None, "selected": False}
        picked = select_clean_think(
            cfg,
            references=extract_references(record.get("user_prompt") or ""),
            candidates=record.get("candidates") or [],
            base_think=base_think,
            pool_answers=(pool_index.get(qid) or {}).get("pool_answers") or [])
        out["chosen_think"] = picked["best_think"]
        out["score_clean"] = picked["score_clean"]
        out["score_base"] = picked["score_base"]
        out["selected"] = picked["selected"]
        return out

    results = gather_until(
        rollouts, _pair,
        enough=lambda rs: target > 0 and sum(1 for r in rs if r and r.get("selected")) >= target,
        chunk=int(cfg.data["gather_chunk"]),
        workers=int(cfg.judge["call_workers"]),
        desc="dpo_pairs",
        progress_path=paths.progress(cfg, "dpo"),
        seed=int(cfg.data["gather_seed"]))

    pairs = [dpo_row(r["user_prompt"], r["chosen_think"], r["base_think"], r["answer"],
                     query=r.get("query"),
                     meta={"score_clean": r.get("score_clean"), "score_base": r.get("score_base")})
             for r in results if r.get("selected")]
    write_jsonl(paths.dpo_pairs(cfg), pairs)
    log.info("偏好对 %d 对 -> %s", len(pairs), paths.dpo_pairs(cfg))
    if 0 < len(pairs) < _MIN_USABLE_PAIRS:
        log.warning("只有 %d 对 < %d：信号太弱。把 gates.n_sigma 调小，或加大 rollout 的 K。",
                    len(pairs), _MIN_USABLE_PAIRS)
    return paths.dpo_pairs(cfg)
