"""产物路径：链路里每个中间文件叫什么、落在哪，集中在这里。

本模块在整条链路里的位置：数据构建和训练编排共用的地址簿。上一步写到哪、下一步从哪读，
两边都调同一个函数，不用各自拼字符串。

命名规则：`NN_名字.jsonl`，两位数字前缀就是它在链路里的顺序。`ls` 一下就能看出跑到哪一步了。
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config


def output_dir(cfg: Config) -> Path:
    """全部中间产物的根目录。"""
    return Path(cfg.paths["output_dir"])


# ---------------------------------------------------------------------------
# 上游：基座重产 → 切分 → 建答案池
# ---------------------------------------------------------------------------

def base_outputs(cfg: Config) -> Path:
    """基座对全部题目重新生成的 think/answer。整条链路的原料。"""
    return output_dir(cfg) / "00_base_outputs.jsonl"


def train_set(cfg: Config) -> Path:
    """训练集。带基座的 think（当选样对照）和 answer（当 answer-lock 锚）。"""
    return output_dir(cfg) / "01_train.jsonl"


def eval_set(cfg: Config) -> Path:
    """冻结验收集。全程不参与任何训练和选样。"""
    return output_dir(cfg) / "01_eval.jsonl"


def problems_all(cfg: Config) -> Path:
    """全量题集（训练 + 验收）。建答案池要覆盖验收题，评测那道答案门才有靶子。"""
    return output_dir(cfg) / "01_problems_all.jsonl"


def problems_train(cfg: Config) -> Path:
    """仅训练题集。自采样只能用它，用全量会把验收题泄漏进训练。"""
    return output_dir(cfg) / "01_problems_train.jsonl"


def answer_pool(cfg: Config) -> Path:
    """基座认可答案池：每题贪心 1 条 + 采样若干条。"""
    return output_dir(cfg) / "02_answer_pool.jsonl"


# ---------------------------------------------------------------------------
# 四个训练阶段各自的数据
# ---------------------------------------------------------------------------

def sft_train(cfg: Config) -> Path:
    """冷启动 SFT 训练集。"""
    return output_dir(cfg) / "10_sft_train.jsonl"


def sft_eval(cfg: Config) -> Path:
    """冷启动留出的验证集。用来选 best checkpoint，绝不进训练桶。"""
    return output_dir(cfg) / "10_sft_eval.jsonl"


def rft_samples(cfg: Config) -> Path:
    """拒绝采样阶段的原始自采样结果，每题 K 条。"""
    return output_dir(cfg) / "20_rft_samples.jsonl"


def rft_train(cfg: Config) -> Path:
    """拒绝采样筛出来的训练集。"""
    return output_dir(cfg) / "21_rft_train.jsonl"


def dpo_rollout(cfg: Config) -> Path:
    """偏好对阶段的原始 rollout，每题 K 条候选。"""
    return output_dir(cfg) / "30_dpo_rollout.jsonl"


def dpo_pairs(cfg: Config) -> Path:
    """构好的偏好对。"""
    return output_dir(cfg) / "31_dpo_pairs.jsonl"


def grpo_data(cfg: Config) -> Path:
    """在线 GRPO 的训练 prompt 集。"""
    return output_dir(cfg) / "40_grpo_prompts.jsonl"


# ---------------------------------------------------------------------------
# 进度文件：贵的步骤边跑边落盘，中断可续
# ---------------------------------------------------------------------------

def progress(cfg: Config, stage: str) -> Path:
    """某个数据构建阶段的进度文件。

    :param cfg: 配置
    :param stage: 阶段名，如 ``sft`` / ``rft`` / ``dpo``
    :return: 进度文件路径
    """
    return output_dir(cfg) / f"progress_{stage}.jsonl"


def stage_marker(cfg: Config, name: str) -> Path:
    """某一步"整步完成"的标记文件。

    为什么不用"输出文件存不存在"来判断：选样一条都没选中是合法结果，会写出一个 0 字节文件；
    而进程被 kill 也会留下一个不完整的文件。两种情况用文件大小分不开，
    所以整步成功才写这个标记，续跑只认它。

    :param cfg: 配置
    :param name: 步骤名
    :return: 标记文件路径
    """
    return output_dir(cfg) / f".{name}.done"


# ---------------------------------------------------------------------------
# 评测产物
# ---------------------------------------------------------------------------

def eval_paths(cfg: Config, tag: str) -> tuple[Path, Path, Path, Path]:
    """某个模型的四份评测产物。

    :param cfg: 配置
    :param tag: 模型标签，如 ``baseline`` / ``sft`` / ``dpo``
    :return: ``(推理结果, 逐条判分, 人读报告, 机读摘要)``
    """
    base = output_dir(cfg) / "eval"
    return (base / f"{tag}_infer.jsonl", base / f"{tag}_scores.jsonl",
            base / f"{tag}_report.md", base / f"{tag}_summary.json")


# ---------------------------------------------------------------------------
# 模型落点
# ---------------------------------------------------------------------------

def lora_dir(cfg: Config, stage: str) -> Path:
    """某阶段的 LoRA adapter 目录。

    :param stage: ``sft`` / ``rft`` / ``dpo`` / ``grpo-warmup`` / ``grpo-online``
    """
    return Path(cfg.paths["ckpt_dir"]) / f"{stage}-lora"


def merged_dir(cfg: Config, stage: str) -> Path:
    """某阶段 LoRA 合并回基座之后的全量模型目录。

    每一份都是一个完整基座大小。磁盘不够就删掉上上个阶段的——下一阶段只依赖上一阶段。
    """
    return Path(cfg.paths["model_dir"]) / f"{stage}-merged"
