"""第 5 步：拒绝采样数据 —— 模型自己采样、自己筛好的、再训自己。

本模块在整条链路里的位置：四阶段的第二阶段。上一阶段的 SFT 让模型学会了"自然推理"这种写法，
但那是从裁判模型的稿子里学的，写法和模型自己的分布有距离；而且冷启动只教写法，
没管 grounding——模型可能学会了不说"参考问答对"，同时也不再老实扣着资料推了。

这一阶段做的事：让**刚训完的模型自己**对训练题采 K 条，用三道门筛出确实更干净的那条，
把它当新的 SFT 目标再训一轮。筛选用的是同一套门（见 :mod:`src.data.selection`），
但答案门在这里格外重要——它把"答案漂了"的候选直接踢掉，等于把 grounding 拉回来。

**拒绝采样和 SFT 的边界。** 训练方式完全一样（都是交叉熵），区别只在数据从哪来：
SFT 的目标是外部给的，拒绝采样的目标是模型自己采出来、再用打分函数筛出来的。
所以它算 RL 的前奏而不是 RL——它用了奖励信号，但没有"根据奖励调整每个 token 的概率"
这一步，只是把高奖励样本当成新的监督目标。

输入：`20_rft_samples.jsonl`（由 :mod:`src.data.rollout` 产出）。
输出：`21_rft_train.jsonl`，格式和冷启动 SFT 完全一样，训练脚本也复用同一个。
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.data import paths
from src.data.build_answer_pool import load_pool_index
from src.data.jsonl_io import gather_until, index_by, read_jsonl, write_jsonl
from src.data.parsing import extract_references
from src.data.schema import sft_row
from src.data.selection import select_clean_think
from src.utils.log import get_logger

log = get_logger("build_rft_data")

# 训练桶少于这个数就告警。模型已经被冷启动洗过一遍，自采的候选大多挤在干净这一端、
# 裁判分不开，产出率天然比冷启动低，所以这个门槛设得比冷启动低。
_MIN_USABLE_SAMPLES = 100


def run(cfg: Config) -> Path:
    """跑第 5 步的筛选部分（采样在 :func:`src.data.rollout.sample_candidates`）。

    :param cfg: 配置
    :return: 训练集路径
    """
    samples = read_jsonl(paths.rft_samples(cfg))
    train_index = index_by(read_jsonl(paths.train_set(cfg)))
    pool_index = load_pool_index(cfg)
    target = int(cfg.train["rft"]["target_samples"])

    log.info("拒绝采样选样：%d 题，目标 %d 条", len(samples), target)

    def _select(record: dict) -> dict:
        qid = record.get("qid")
        base = train_index.get(qid) or {}
        out = {"qid": qid, "query": record.get("query"),
               "user_prompt": record.get("user_prompt"),
               "answer": (base.get("answer") or "").strip(),
               "best_think": None, "selected": False}
        picked = select_clean_think(
            cfg,
            references=extract_references(record.get("user_prompt") or ""),
            candidates=record.get("candidates") or [],
            base_think=(base.get("reasoning") or "").strip(),
            pool_answers=(pool_index.get(qid) or {}).get("pool_answers") or [])
        out.update({k: picked[k] for k in ("best_think", "score_clean", "score_base", "selected")})
        return out

    results = gather_until(
        samples, _select,
        enough=lambda rs: target > 0 and sum(1 for r in rs if r and r.get("selected")) >= target,
        chunk=int(cfg.data["gather_chunk"]),
        workers=int(cfg.judge["call_workers"]),
        desc="rft_data",
        progress_path=paths.progress(cfg, "rft"),
        seed=int(cfg.data["gather_seed"]))

    train_rows = [sft_row(r["user_prompt"], r["best_think"], r["answer"], query=r.get("query"))
                  for r in results if r.get("selected")]
    write_jsonl(paths.rft_train(cfg), train_rows)
    log.info("拒绝采样选中 %d 条 -> %s", len(train_rows), paths.rft_train(cfg))
    if 0 < len(train_rows) < _MIN_USABLE_SAMPLES:
        log.warning("训练桶只有 %d 条 < %d：模型自采已普遍偏干净、裁判分辨率到顶，属于预期。",
                    len(train_rows), _MIN_USABLE_SAMPLES)
    return paths.rft_train(cfg)
