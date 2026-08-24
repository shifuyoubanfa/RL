"""第 2 步：切训练集和冻结验收集，并产出建池用的题集。

本模块在整条链路里的位置：紧接第 1 步。它决定"哪些题这辈子都不许进训练"。

输入：`00_base_outputs.jsonl`。
输出四份：

- `01_train.jsonl` —— 训练集，带基座 think 和 answer。
- `01_eval.jsonl` —— 冻结验收集，**固定条数**（不是比例），全程不参与训练和选样。
- `01_problems_all.jsonl` —— 全量题集，给建答案池用。验收题也要有池，否则评测那道答案门没有靶子。
- `01_problems_train.jsonl` —— 仅训练题集，给自采样用。

**用固定条数而不是比例。** 语料条数会因为上游丢弃空答案而小幅波动，按比例切会让验收集
每次跑都差几条——两次评测的分数就不能直接比了。固定 500 条 + 固定种子，验收集永远是同一批题。

**验收集必须在任何训练动作之前切出来。** 顺序反了就是数据泄漏，而且是查不出来的那种：
指标会好看，但好看的原因是模型见过题。
"""

from __future__ import annotations

import random
from pathlib import Path

from src.config import Config
from src.data import paths
from src.data.jsonl_io import qid_of, read_jsonl, write_jsonl
from src.utils.log import get_logger

log = get_logger("split_dataset")

# 有效样本低于这个数就告警。语料在千条量级，掉到四位数以下基本说明上一步被中断了。
_SUSPICIOUS_FLOOR = 1000


def run(cfg: Config) -> dict[str, Path]:
    """跑第 2 步。

    :param cfg: 配置
    :return: ``{train, eval, problems_all, problems_train}`` 四个路径
    :raise SystemExit: 有效样本还不够切出验收集
    """
    n_eval = int(cfg.data["n_eval"])
    seed = int(cfg.data["split_seed"])
    records = read_jsonl(paths.base_outputs(cfg))

    # 1.【清洗】丢掉没有参考资料或没有答案的，并按 qid 去重
    rows: list[dict] = []
    seen: set[str] = set()
    for record in records:
        query = record.get("query")
        user_prompt = (record.get("user_prompt") or "").strip()
        answer = (record.get("answer") or "").strip()
        think = (record.get("reasoning") or "").strip()
        if not user_prompt or not answer:
            continue
        qid = qid_of(query)
        if qid in seen:                                # 同题重复出现，只留第一条
            continue
        seen.add(qid)
        rows.append({"qid": qid, "query": query, "user_prompt": user_prompt,
                     "answer": answer, "reasoning": think})
    log.info("有效样本 %d / %d", len(rows), len(records))
    if len(rows) <= n_eval:
        raise SystemExit(f"有效样本 {len(rows)} <= 验收集 {n_eval}，切不出来")

    # 2.【切分】固定种子洗牌，前 n_eval 条当验收集
    random.Random(seed).shuffle(rows)
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    for row in eval_rows:
        row["split"] = "eval"
    for row in train_rows:
        row["split"] = "train"

    write_jsonl(paths.eval_set(cfg), eval_rows)
    write_jsonl(paths.train_set(cfg), train_rows)

    # 3.【题集】建池用。全量那份含验收题——评测的答案门要拿验收题的池子当靶子。
    problems = [{"qid": r["qid"], "split": r["split"], "query": r["query"],
                 "user_prompt": r["user_prompt"], "gold_answer": r["answer"]}
                for r in (eval_rows + train_rows)]
    write_jsonl(paths.problems_all(cfg), problems)
    write_jsonl(paths.problems_train(cfg), [p for p in problems if p["split"] == "train"])

    log.info("train=%d -> %s", len(train_rows), paths.train_set(cfg))
    log.info("eval =%d（冻结）-> %s", len(eval_rows), paths.eval_set(cfg))
    if len(rows) < _SUSPICIOUS_FLOOR:
        log.warning("有效样本只有 %d 条：上一步可能被中断，产物不完整，核对后再往下跑。", len(rows))

    return {"train": paths.train_set(cfg), "eval": paths.eval_set(cfg),
            "problems_all": paths.problems_all(cfg), "problems_train": paths.problems_train(cfg)}
