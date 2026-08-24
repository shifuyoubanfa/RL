"""Rollout：让当前模型对每道训练题采 K 条候选。

本模块在整条链路里的位置：拒绝采样和偏好对两个阶段共用的采样层。GRPO 阶段不走这里——
它的采样发生在训练进程内部，由训练框架带的推理引擎现场做。

**什么是 rollout，为什么强化学习非要它。** 监督微调的训练数据是现成的：一条输入配一条
标准答案，算交叉熵就完了。强化学习没有标准答案，只有一个"这条输出好不好"的打分函数。
要知道模型现在写得怎么样、哪条写法更好，只能让它**自己先写出来**，再拿去打分——
这个"自己先写一批"就是 rollout。所以 RL 的每一步都比 SFT 多一轮完整的生成开销。

采样温度必须高于 0。贪心采出来的 K 条几乎一模一样，组内没有差异，
"哪条更好"这个信号就不存在了，后面的筛选和优势估计都无从谈起。
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.data.jsonl_io import qid_of, read_jsonl, write_jsonl
from src.data.prompts import NEUTRAL_SYSTEM_PROMPT
from src.models import vllm_client
from src.utils.concurrency import map_concurrent
from src.utils.log import get_logger

log = get_logger("rollout")


def sample_candidates(cfg: Config, *, problems_path: Path, out_path: Path,
                      model_name: str, k: int, max_tokens: int | None = None) -> Path:
    """对题集里每道题采 K 条候选，写成 jsonl。

    system 提示用中性那套，不用基座的 RAG 腔——这里采样的是**训练后的模型**，
    给它套回 RAG 腔提示等于把刚洗掉的腔调用提示词又拉回来，采出来的候选全带检索腔。

    :param cfg: 配置
    :param problems_path: 题集文件，每行需含 ``qid``、``user_prompt``
    :param out_path: 输出路径
    :param model_name: 推理服务里该模型的名字
    :param k: 每题采几条
    :param max_tokens: 单条最大长度，不给就用配置里的 rollout 上限
    :return: 输出路径
    """
    data_cfg = cfg.data
    problems = read_jsonl(problems_path)
    max_tokens = int(max_tokens or data_cfg["rollout_max_tokens"])

    vllm_client.wait_ready()
    log.info("rollout：model=%s 题数=%d K=%d 温度=%.2f",
             model_name, len(problems), k, float(data_cfg["rollout_temperature"]))

    def _roll(problem: dict) -> dict | None:
        user_prompt = problem.get("user_prompt") or ""
        try:
            candidates = vllm_client.generate_k(
                user_prompt, k=k, model=model_name, system=NEUTRAL_SYSTEM_PROMPT,
                temperature=float(data_cfg["rollout_temperature"]),
                top_p=float(data_cfg["rollout_top_p"]), max_tokens=max_tokens)
        except Exception as exc:                       # noqa: BLE001 - 单题失败降级成"这题没候选"，总数日志里看得到
            log.warning("rollout 失败 qid=%s: %r", problem.get("qid"), exc)
            return None
        return {
            "qid": problem.get("qid") or qid_of(problem.get("query") or ""),
            "query": problem.get("query"),
            "user_prompt": user_prompt,
            "gold_answer": problem.get("gold_answer") or problem.get("answer") or "",
            "candidates": candidates,
        }

    results = [r for r in map_concurrent(problems, _roll,
                                         workers=int(cfg.vllm["call_workers"]),
                                         desc=f"rollout:{model_name}") if r]
    write_jsonl(out_path, results)
    log.info("完成：%d 题 × K=%d -> %s", len(results), k, out_path)
    return out_path
