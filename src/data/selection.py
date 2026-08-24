"""候选筛选：从一题的 K 条 rollout 里挑出一条"确实更干净"的推理，挑不出就整题弃。

本模块在整条链路里的位置：拒绝采样和偏好对两个阶段共用的选样逻辑。两个阶段筛出来的东西
用途不同（一个当 SFT 目标，一个当偏好对的 chosen），但"什么样的候选算好"这个判据必须是同一套，
所以只在这里实现一次。

**三道门，顺序是成本决定的：**

1. **答案门**：候选的 answer 漂出基座认可池就丢。确定性判断。丢它不是因为答案错了，
   而是因为它的 think 正朝一个错结论推——把这种推理教给模型，等于教它推错方向。
2. **规则门**：候选的 think 还带检索腔就丢。确定性判断。
3. **σ-可分门**：活下来的候选先各打 `k_screen` 遍，取最高的那一条；再对这一条打 `k_select` 遍，
   确认它比基座原推理干净到误差带不相交。

**第 3 步里有两个不同的 argmax 语义，别混。** 粗筛那一次是在候选之间挑最好的一条，
不和基座比；确认那一次才是和基座比、判可不可分。写在一起会让人以为粗筛也带了可分判据，
实际上粗筛只是"挑一条最有希望的去做贵的确认"。
"""

from __future__ import annotations

from src.config import Config
from src.data.parsing import parse_think_answer
from src.rewards.calibration import confident_cleaner
from src.rewards.judge import score_for_select, score_think
from src.rewards.rules import answer_in_v1_pool, detect_rag_style


def select_clean_think(cfg: Config, *, references: str, candidates: list[str],
                       base_think: str, pool_answers: list[str]) -> dict:
    """从一题的候选里挑出一条更干净的推理。

    :param cfg: 配置
    :param references: 该题的参考资料
    :param candidates: K 条 rollout 原文
    :param base_think: 基座原推理，当"脏"对照
    :param pool_answers: 基座认可池答案
    :return: ``{best_think, score_clean, score_base, selected, n_survivors}``；
             `selected` 为 True 才算这一题产出了样本
    """
    gates = cfg.gates
    out = {"best_think": None, "score_clean": None, "score_base": None,
           "selected": False, "n_survivors": 0}
    if not base_think:
        return out

    # 1.【答案门 + 规则门】两道确定性的门先过一遍，零成本
    survivors: list[str] = []
    for text in candidates or []:
        think, answer = parse_think_answer(text)
        if not think.strip():
            continue
        if not answer_in_v1_pool(answer, pool_answers)["in_pool"]:
            continue
        if detect_rag_style(think)["has_rag_style"]:
            continue
        survivors.append(think)
    out["n_survivors"] = len(survivors)
    if not survivors:
        return out

    # 2.【粗筛挑一条】在候选之间取分最高的。这里是 argmax，不带可分判据
    k_screen = int(gates["k_screen"])
    scored = [(t, score_think(references, t, k_screen)["clean_score"]) for t in survivors]
    scored = [(t, s) for t, s in scored if s is not None]
    if not scored:
        return out
    best_think = max(scored, key=lambda pair: pair[1])[0]

    # 3.【确认可分】只对选中的这一条上贵的口径，和基座原推理比误差带
    k_select = int(gates["k_select"])
    score_clean = score_for_select(references, best_think, k_select)["clean_score"]
    score_base = score_for_select(references, base_think, k_select)["clean_score"]
    if score_clean is None or score_base is None:
        return out

    out["best_think"] = best_think
    out["score_clean"] = score_clean
    out["score_base"] = score_base
    out["selected"] = confident_cleaner(score_clean, score_base, float(gates["n_sigma"]))
    return out
