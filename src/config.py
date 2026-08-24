"""配置加载：把 `configs/train.yaml` 读成一个可以点属性取值的对象。

本模块在整条链路里的位置：最上游。数据构建、训练编排、评测三个入口进程一启动就调
`load_config()`，之后全链路任何一处要路径、要超参、要门限，都从这里取，不再各自写默认值。

三件事：

1. **读 YAML**，缺文件就报错，不静默用内置默认——静默默认会让"我改了配置怎么没生效"变成
   一个查半天的问题。
2. **展开 `${work_dir}` 这类占位符**，让 yaml 里能写相对引用，不必把根目录抄五遍。
3. **环境变量覆盖**：`RL_WORK_DIR`、`RL_BASE_MODEL` 这些覆盖同名配置项。CI、临时换机器、
   一次性调小样本量都靠它，不用改文件。

健壮性：本模块不建目录、不碰网络、不读权重。它唯一的副作用是缓存一份已解析的配置
（`_CACHED`），让同一进程里反复 `load_config()` 不重复解析 YAML。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

# 仓库根目录 = 本文件的上两级。全链路所有相对路径都以它为基准，
# 这样从任何工作目录启动脚本，行为都一样。
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "train.yaml"

# 环境变量 -> 配置项路径。只覆盖这些；其余想改就改 yaml。
# 挑选标准：换一台机器就必须改的（路径、卡号、模型位置），以及联调时想临时调小的（样本量）。
_ENV_OVERRIDES: dict[str, str] = {
    "RL_WORK_DIR": "paths.work_dir",
    "RL_OUTPUT_DIR": "paths.output_dir",
    "RL_CKPT_DIR": "paths.ckpt_dir",
    "RL_MODEL_DIR": "paths.model_dir",
    "RL_LOG_DIR": "paths.log_dir",
    "RL_RAW_CORPUS": "paths.raw_corpus",
    "RL_BASE_MODEL": "model.base_model",
    "RL_VLLM_BASE_URL": "vllm.base_url",
    "RL_VLLM_GPUS": "vllm.gpus",
    "RL_TRAIN_GPUS": "train.gpus",
    "RL_JUDGE_MODEL": "judge.model",
    "RL_JUDGE_BUDGET_YUAN": "judge.budget_yuan",
    "RL_N_EVAL": "data.n_eval",
    "RL_SFT_TARGET": "train.sft.target_samples",
    "RL_RFT_TARGET": "train.rft.target_samples",
    "RL_DPO_TARGET": "train.dpo.target_pairs",
    "RL_GRPO_WARMUP_STEPS": "train.grpo.warmup.steps",
    "RL_GRPO_ONLINE_STEPS": "train.grpo.online.steps",
}

_PLACEHOLDER_RE = re.compile(r"\$\{([a-zA-Z0-9_]+)\}")

_CACHED: Config | None = None


class Config(dict):
    """一份已解析的配置。既是 dict，也支持点号取值。

    点号取值只是为了让调用处读起来顺（`cfg.train.sft.epochs`），底层仍是普通 dict，
    可以直接 `json.dumps` 落进日志。
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:                       # 拼错 key 要当场炸，不能返回 None 往下传
            raise AttributeError(f"配置里没有 {name!r}；现有键：{sorted(self.keys())}") from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """按 `a.b.c` 取深层值，取不到返回 default。

        :param dotted: 点号分隔的配置路径，例如 ``train.sft.epochs``
        :param default: 路径不存在时返回什么
        :return: 该路径上的值；中途遇到非 dict 也算不存在
        """
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _set_path(tree: dict, dotted: str, value: Any) -> None:
    """按 `a.b.c` 写深层值，路径上缺的层自动补成 dict。"""
    node = tree
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _coerce(raw: str, old: Any) -> Any:
    """把环境变量里的字符串对齐到原配置项的类型。

    环境变量只有字符串。如果原值是 int/float/bool，就按原类型转一次；
    转不动的原样当字符串用——宁可让下游拿到一个明显不对的字符串报错，
    也不要在这里吞掉、让人以为覆盖生效了。

    :param raw: 环境变量的原始字符串
    :param old: yaml 里原来的值，只用来看类型
    :return: 转换后的值
    """
    if isinstance(old, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    for caster in (int, float):
        if isinstance(old, caster):
            try:
                return caster(raw)
            except ValueError:                        # noqa: PERF203 - 转不动就退回字符串，让下游报清晰的错
                return raw
    return raw


def _expand_placeholders(tree: dict) -> None:
    """就地展开 `paths` 段里的 `${key}` 引用。

    只在 `paths` 段内解析，且只引用同段内已经解析好的键。这样限制是故意的：
    支持全局任意引用会让"这个路径最终指到哪"变成要跨段追的问题。

    :param tree: 完整配置树，会被就地修改
    """
    paths = tree.get("paths")
    if not isinstance(paths, dict):
        return
    resolved: dict[str, str] = {}
    for key, value in paths.items():
        if not isinstance(value, str):
            resolved[key] = value
            continue
        # 逐个替换 ${x}；引用了尚未解析的键就原样留着，让下游路径显式报错而不是拼出个半截路径
        paths[key] = _PLACEHOLDER_RE.sub(
            lambda m: str(resolved.get(m.group(1), m.group(0))), value)
        resolved[key] = paths[key]


def load_config(path: str | Path | None = None, *, reload: bool = False) -> Config:
    """读配置文件，应用环境变量覆盖，返回可点号取值的配置对象。

    :param path: 配置文件路径；不给就用 `configs/train.yaml`，也可用 `RL_CONFIG` 指定
    :param reload: True 则忽略进程内缓存重新解析（测试里改了 yaml 想立刻生效时用）
    :return: 解析好的 :class:`Config`
    :raise FileNotFoundError: 配置文件不存在
    :raise ValueError: YAML 顶层不是字典
    """
    global _CACHED
    if _CACHED is not None and path is None and not reload:
        return _CACHED

    cfg_path = Path(path or os.environ.get("RL_CONFIG") or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件 {cfg_path}；用 --config 或 RL_CONFIG 指定")
    tree = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(tree, dict):
        raise ValueError(f"{cfg_path} 顶层必须是字典")

    # 1.【覆盖】环境变量优先级高于文件。先覆盖再展开占位符，
    #    这样 RL_WORK_DIR 一改，下面四个派生目录会跟着改。
    for env_name, dotted in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        _set_path(tree, dotted, _coerce(raw, Config(tree).get_path(dotted)))

    # 2.【展开】把 paths 段里的 ${work_dir} 换成真值
    _expand_placeholders(tree)

    cfg = Config(tree)
    if path is None:
        _CACHED = cfg
    return cfg


def ensure_dirs(cfg: Config) -> None:
    """按配置把产物目录建出来。

    只在真正要写东西的入口（数据构建 / 训练 / 评测）调一次。纯读配置的地方
    （比如单元测试、`--help`）不该有建目录这种副作用。

    :param cfg: 已加载的配置
    """
    for key in ("output_dir", "ckpt_dir", "model_dir", "log_dir"):
        Path(cfg.paths[key]).mkdir(parents=True, exist_ok=True)


def judge_api_key(cfg: Config) -> str:
    """取裁判服务的 API key。

    key 只从环境变量读，代码里不留任何兜底常量——本仓库是公开的，一旦写进来就等于泄露。

    :param cfg: 已加载的配置
    :return: key 字符串
    :raise RuntimeError: 环境变量没设
    """
    env_name = cfg.judge["api_key_env"]
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise RuntimeError(
            f"缺少环境变量 {env_name}。先 `export {env_name}=<your-key>`，"
            f"或参考 .env.example 配好再跑。")
    return key
