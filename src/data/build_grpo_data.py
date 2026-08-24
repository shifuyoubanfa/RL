"""第 7 步：构在线 GRPO 的训练 prompt 集。

本模块在整条链路里的位置：四阶段的最后一阶段的输入。

和前面三步不同，这一步产出的行里**没有答案**——GRPO 的答案是训练时现场采出来的。
行里带的是"采完之后算奖励要用的料"：题面，以及基座认可池。

**三道拒绝，全是硬失败，不是警告：**

1. **验收题混进来** → 直接退出。在线训练会把这道题反复采几十次，一旦泄漏，
   后面所有评测数字都不作数，而且看不出来。
2. **某题缺答案池** → 直接退出。奖励函数里的答案硬门要拿池当靶子，没有池就没法判，
   静默跳过等于让一部分训练题绕开了硬门。
3. **过滤后一条都不剩** → 直接退出。

**第三类过滤是警告不是失败**：答案池里抽不出任何极性、数字、日期的题会被跳过。
这类题的池子给不出可比对的靶子，模型答什么都判不了漂——留着它们，模型会学到
"少说具体数字就能安全拿分"，正好把 grounding 反向优化掉。跳过的条数会打进日志。
"""

from __future__ import annotations

import random
from pathlib import Path

from src.config import Config
from src.data import paths
from src.data.build_answer_pool import load_pool_index, pool_answers_of
from src.data.jsonl_io import qid_of, read_jsonl, write_jsonl
from src.data.schema import grpo_row
from src.rewards.rules import answer_in_v1_pool
from src.utils.log import get_logger

log = get_logger("build_grpo_data")

# 报错信息里最多列几个出问题的 qid。够定位，又不会把日志刷满。
_MAX_REPORTED_QIDS = 8


def pool_is_trainable(pool_answers: list[str]) -> bool:
    """这个答案池能不能支撑在线的答案硬门。

    判据：池里至少有一条答案，拿它自己去比这个池，既 `in_pool` 又 `comparable`。
    换句话说，池里得真有可抽取的极性/数字/日期，硬门才有东西可比。

    :param pool_answers: 该题的池答案
    :return: 可训练为 True
    """
    for answer in pool_answers:
        verdict = answer_in_v1_pool(answer, pool_answers)
        if verdict.get("in_pool") and verdict.get("comparable", True):
            return True
    return False


def build_rows(train_rows: list[dict], pool_index: dict[str, dict], *,
               eval_qids: set[str] | None = None,
               shuffle_seed: int | None = 42, limit: int = 0) -> list[dict]:
    """把训练题集 + 答案池组装成 GRPO 行。

    :param train_rows: 训练题集，每行须含 ``qid``/``query``/``user_prompt``
    :param pool_index: qid 到答案池记录的映射
    :param eval_qids: 冻结验收集的 qid，撞上就报错
    :param shuffle_seed: 洗牌种子；None = 保持原顺序
    :param limit: 洗牌后只留前 N 条，0 = 全量。联调用
    :return: GRPO 行列表
    :raise SystemExit: 混入验收题、有题缺池、或过滤后为空
    """
    pool = list(train_rows)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(pool)
    if limit and limit > 0:
        pool = pool[:limit]

    rows: list[dict] = []
    leaked: list[str] = []
    missing: list[str] = []
    untrainable: list[str] = []

    for record in pool:
        query = record.get("query") or ""
        qid = record.get("qid") or qid_of(query)
        split = record.get("split")
        if (split and split != "train") or (eval_qids and qid in eval_qids):
            leaked.append(qid)
            continue

        pool_answers = pool_answers_of(pool_index.get(qid) or {})
        if not pool_answers:
            missing.append(qid)
            continue
        if not pool_is_trainable(pool_answers):
            untrainable.append(qid)
            continue

        gold = (record.get("gold_answer") or record.get("answer")
                or (pool_index.get(qid) or {}).get("canonical_answer") or "")
        rows.append(grpo_row(qid, query, record.get("user_prompt") or "", gold, pool_answers))

    if leaked:
        raise SystemExit(f"GRPO 数据混入了非训练集的题，拒绝继续。示例 qid={leaked[:5]}")
    if missing:
        raise SystemExit(f"有 {len(missing)} 道题缺答案池，拒绝在线训练；先补第 3 步。"
                         f"示例 qid={missing[:_MAX_REPORTED_QIDS]}")
    if untrainable:
        log.warning("跳过 %d 道答案池抽不出可比事实的题；示例 qid=%s",
                    len(untrainable), untrainable[:_MAX_REPORTED_QIDS])
    if not rows:
        raise SystemExit("GRPO 数据过滤后为空：所有训练题的答案池都不可用，拒绝继续。")
    return rows


def run(cfg: Config, *, limit: int = 0) -> Path:
    """跑第 7 步。

    :param cfg: 配置
    :param limit: 只留前 N 条，0 = 全量
    :return: 输出路径
    :raise SystemExit: 上游产物缺失，或组装时触发任一硬失败
    """
    train_rows = read_jsonl(paths.problems_train(cfg))
    pool_index = load_pool_index(cfg)
    eval_rows = read_jsonl(paths.eval_set(cfg))
    if not train_rows:
        raise SystemExit(f"训练题集为空: {paths.problems_train(cfg)}")
    if not pool_index:
        raise SystemExit(f"答案池为空: {paths.answer_pool(cfg)}")

    eval_qids = {r.get("qid") or qid_of(r.get("query") or "") for r in eval_rows
                 if r.get("qid") or r.get("query")}
    rows = build_rows(train_rows, pool_index, eval_qids=eval_qids,
                      shuffle_seed=int(cfg.data["gather_seed"]), limit=limit)
    write_jsonl(paths.grpo_data(cfg), rows)
    log.info("GRPO 训练 prompt %d 条 -> %s", len(rows), paths.grpo_data(cfg))
    return paths.grpo_data(cfg)
