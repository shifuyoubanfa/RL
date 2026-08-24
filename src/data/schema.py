"""训练样本的拼装：SFT 行、DPO 偏好对行、GRPO prompt 行。

本模块在整条链路里的位置：数据构建的出口。三种训练样本格式只在这里拼，其它地方一律调用，
不许自己手写 dict——格式漂一点，ms-swift 那边要么直接报错，要么更糟：按默认值静默兜底。

**贯穿三种格式的一条铁律：answer-lock。**

训练样本里的 `<answer>` 永远拼基座自己那一版原文，只有 `<think>` 不同。

为什么必须这样。这条链路的目标是"把推理过程的写法洗干净，同时答案一个字不动"。
一个更省事的做法是：让裁判模型把 think 和 answer 一起重写，反正它写得更漂亮。
问题在于梯度不认识"我只想改 think"这件事——answer 段一旦跟着变，模型学到的就是
"整段输出都可以改"，几百步之后 answer 开始漂，而 think 干净分还在涨，指标全绿、
产品坏掉。answer 段逐字锁死原文，chosen 和 rejected 在 answer 段完全相同，
DPO 的对数比在那一段天然抵消，梯度只能压到 think 上。

副作用是偏好对的构造变得**零成本**：rejected 直接用基座原推理配同一段原答案，
不需要额外生成、额外打分。而且这一对天然分得开——基座原推理就是满是检索腔的那一版。
"""

from __future__ import annotations

import json

from src.data.parsing import wrap_completion
from src.data.prompts import NEUTRAL_SYSTEM_PROMPT


def sft_row(user_prompt: str, think: str, answer: str, query: str | None = None) -> dict:
    """一条 SFT 训练样本。冷启动和拒绝采样两个阶段共用这个格式。

    system 用中性提示，不用基座那套 RAG 腔——训练数据里烤进什么提示词，推理时就得配什么提示词。

    :param user_prompt: 题面（含参考资料）
    :param think: 要教给模型的推理段
    :param answer: 基座原答案，逐字不动
    :param query: 题面里的问题本身，只作溯源用，训练不读
    :return: ms-swift SFT 格式的一行
    """
    return {
        "messages": [
            {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt or ""},
            {"role": "assistant", "content": wrap_completion(think, answer)},
        ],
        "query": query,
    }


def dpo_row(user_prompt: str, chosen_think: str, rejected_think: str, answer: str,
            query: str | None = None, meta: dict | None = None) -> dict:
    """一条 DPO 偏好对。

    ms-swift 的偏好对格式是：chosen 放在 `messages` 最后一条 assistant 里，
    rejected 放顶层 `rejected_response` 字段。

    两边共用**同一段** answer，只有 think 不同。见模块开头的 answer-lock。

    :param user_prompt: 题面
    :param chosen_think: 洗干净的推理段
    :param rejected_think: 基座原推理段
    :param answer: 基座原答案，chosen 和 rejected 共用
    :param query: 溯源用
    :param meta: 附带的选样分数等，训练不读，出问题时用来回查这一对是怎么选中的
    :return: ms-swift DPO 格式的一行
    """
    return {
        "messages": [
            {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt or ""},
            {"role": "assistant", "content": wrap_completion(chosen_think, answer)},
        ],
        "rejected_response": wrap_completion(rejected_think, answer),
        "query": query,
        "meta": meta or {},
    }


def grpo_row(qid: str, query: str, user_prompt: str, gold_answer: str,
             pool_answers: list[str]) -> dict:
    """一条 GRPO 训练 prompt。

    和前两种不同，GRPO 行**没有 assistant 段**——答案是训练时现场采出来的。
    行里带的是采完之后算奖励要用的那些料。

    `pool_answers` 以 JSON 字符串的形式随行带上，而不是让奖励函数去读一个共享文件：
    GRPO 的奖励函数跑在多个分布式 worker 里，让每个 worker 各自去读一个外部路径，
    等于给"某个 worker 读到的是旧文件"留了口子，而这种错完全不报错，只是奖励算歪。

    :param qid: 题目主键
    :param query: 问题本身
    :param user_prompt: 题面
    :param gold_answer: 基座原答案
    :param pool_answers: 基座认可池里的全部答案
    :return: ms-swift GRPO 格式的一行
    """
    return {
        "messages": [
            {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt or ""},
        ],
        "qid": qid,
        "query": query,
        "user_prompt": user_prompt,
        "gold_answer": gold_answer,
        "v1_answers_json": json.dumps(pool_answers, ensure_ascii=False),
        "pool_size": len(pool_answers),
    }
