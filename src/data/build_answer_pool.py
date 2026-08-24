"""第 3 步：建"基座认可答案池"。

本模块在整条链路里的位置：给后面所有阶段提供**答案漂移的判定靶子**。

做法：让基座对每道题贪心生成 1 条 + 高温采样 N 条，把这 N+1 条答案存下来。

**为什么要采多条，而不是拿贪心那一条当标准。** 一个更省事的做法是：把贪心答案当唯一正确答案，
模型答得和它不一样就算漂。这条路会把基座自己的正常波动也算成漂——同一道题，
基座这次说"免征增值税"，下次说"可以免税"，措辞不同、意思一样。拿单条当锚，
模型只要换个说法就被判失败，训练信号里混进大量假阳性。

采 N 条，取的是"基座自己愿意给出的答案**范围**"。模型答案里的极性和数字只要落在这个范围内，
就算没漂。判定逻辑见 :func:`src.rewards.rules.answer_in_v1_pool`。

**这层挡不住什么**：基座本身答错的题。基座错了，模型跟着错也算没漂。这是刻意的——
这条链路不改答案对错，只改推理的写法。想提高准确率是另一件事，不在这条链路里做。

基座必须用它原生的 RAG 腔 system 提示，才能还原它真实的答案分布。
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.data import paths
from src.data.jsonl_io import read_jsonl, write_jsonl
from src.data.parsing import parse_think_answer
from src.data.prompts import RAG_SYSTEM_PROMPT
from src.models import vllm_client
from src.utils.concurrency import map_concurrent
from src.utils.log import get_logger

log = get_logger("build_answer_pool")


def run(cfg: Config) -> Path:
    """跑第 3 步。

    :param cfg: 配置
    :return: 答案池文件路径
    """
    data_cfg = cfg.data
    n_samples = int(data_cfg["n_pool_samples"])
    problems = read_jsonl(paths.problems_all(cfg))
    out_path = paths.answer_pool(cfg)

    vllm_client.wait_ready()
    log.info("建答案池：%d 道题 × (1 贪心 + %d 采样)", len(problems), n_samples)

    def _build(problem: dict) -> dict | None:
        user_prompt = problem["user_prompt"]
        try:
            canonical = vllm_client.generate_one(
                user_prompt, system=RAG_SYSTEM_PROMPT, temperature=0.0,
                max_tokens=int(data_cfg["build_max_tokens"]))
            samples = vllm_client.generate_k(
                user_prompt, k=n_samples, system=RAG_SYSTEM_PROMPT,
                temperature=float(data_cfg["pool_temperature"]),
                top_p=float(data_cfg["pool_top_p"]),
                max_tokens=int(data_cfg["build_max_tokens"]))
        except Exception as exc:                       # noqa: BLE001 - 单题采样失败降级成"这题没池"，由下面的覆盖率检查暴露
            log.warning("采样失败 qid=%s: %r", problem["qid"], exc)
            return None

        canonical_think, canonical_answer = parse_think_answer(canonical)
        return {
            "qid": problem["qid"], "split": problem["split"], "query": problem["query"],
            "user_prompt": user_prompt, "gold_answer": problem["gold_answer"],
            "canonical_think": canonical_think, "canonical_answer": canonical_answer,
            "pool_answers": [parse_think_answer(s)[1] for s in samples],
        }

    results = [r for r in map_concurrent(problems, _build,
                                         workers=int(cfg.vllm["call_workers"]),
                                         desc="answer_pool") if r]
    write_jsonl(out_path, results)

    # 覆盖率检查：缺池的题会在答案门被一律判成漂移，既污染训练选样，也污染评测分母。
    # 缺口大就把这一步的完成标记删掉重建，别带着缺口往下跑。
    if len(results) < len(problems):
        log.warning("答案池覆盖 %d/%d，缺 %d 题。缺池题会在答案门被判漂，"
                    "缺口大就删掉这一步的完成标记重建。",
                    len(results), len(problems), len(problems) - len(results))
    log.info("答案池完成 %d 题 -> %s", len(results), out_path)
    return out_path


def load_pool_index(cfg: Config, path: str | Path | None = None) -> dict[str, dict]:
    """把答案池读成 ``{qid: 记录}``。

    :param cfg: 配置
    :param path: 覆盖默认路径
    :return: qid 到池记录的映射
    """
    from src.data.jsonl_io import index_by
    return index_by(read_jsonl(path or paths.answer_pool(cfg)))


def pool_answers_of(pool_record: dict) -> list[str]:
    """从一条池记录里取出全部候选答案，去重且保序。

    三个来源按优先级拼起来：贪心答案、语料自带的金标准、N 条采样答案。
    贪心排最前，因为下游有些地方要"取一条有代表性的答案"，取到贪心那条最合适。

    :param pool_record: 一条池记录
    :return: 去重后的答案列表
    """
    raw = [pool_record.get("canonical_answer"), pool_record.get("gold_answer"),
           *(pool_record.get("pool_answers") or [])]
    seen: set[str] = set()
    answers: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        answers.append(text)
    return answers
