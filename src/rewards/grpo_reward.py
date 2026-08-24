"""在线 GRPO 的奖励函数，以 ms-swift 外部插件的形式注册。

本模块在整条链路里的位置：在线阶段的心脏。训练框架每采出一组候选，就调这里给每条打一个标量分，
框架拿这些分算组内优势，再反传到每个 token 上。

ms-swift 的用法：

    swift rlhf --rlhf_type grpo --external_plugins src/rewards/grpo_reward.py \\
               --reward_funcs online ...

**奖励是词典序硬门，不是加权和。**

    格式失败  <  答案漂出池  <  答案没有可比事实  <  规则查出检索腔  <  规则也过了

越靠前越不可补偿。一段写得再漂亮、裁判打满分的 think，只要答案漂了，就永远拿不到
比"规则失败但答案合格"更高的分。

**一个更省事的做法是加权和**：`reward = 0.5×答案分 + 0.3×规则分 + 0.2×裁判分`。
这条路会静默塌方：裁判分的取值范围比另外两项宽，模型很快会发现"把 think 写得极其漂亮
可以补偿答案上的一点漂移"，于是开始一边把 think 洗得越来越干净，一边慢慢改答案。
三个指标里两个在涨，最后那个跌得慢，看曲线一切正常——直到有人去读输出，才发现答案变了。
硬门把这条路堵死：答案漂了就是负分，裁判分连加都不加。

**裁判只在安全区内排序。** 只有格式、答案、规则三道门全过的候选才会去调裁判，
拿到的分只用来在这批"都合格"的候选之间分高下。这既省调用，也保证裁判的噪声
永远不会把一个不合格的候选顶到合格候选之上。

**它挡不住什么。** 硬门管的是"别把答案改了"和"别有表面检索腔"。模型仍然可以在这两条
之内钻空子：把 think 写得又臭又长凑字数，或者反复念同一句话。所以另外挂了两个惩罚项——
重复片段惩罚和超长惩罚，见 :func:`_repeat_penalty` 和 :func:`_length_penalty`。
这两项是补丁不是护栏，真正的长度和重复问题要靠评测时人去读样本。
"""

from __future__ import annotations

import contextlib
import json
import re
import statistics
import sys
from pathlib import Path

# ms-swift 用 --external_plugins 按文件路径加载本模块，此时仓库根目录不在 sys.path 上，
# 下面这些 src.* 的 import 会失败。补进去。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import load_config                                        # noqa: E402
from src.data.parsing import extract_references, parse_think_answer_diagnostic  # noqa: E402
from src.rewards import budget                                            # noqa: E402
from src.rewards.judge import judge_clean_score                           # noqa: E402
from src.rewards.rules import answer_in_v1_pool, detect_rag_style         # noqa: E402

# ---------------------------------------------------------------------------
# 接上 ms-swift 的 ORM 注册表
# ---------------------------------------------------------------------------
# 只捕 ModuleNotFoundError，绝不用裸 except：否则"装了 swift 但导入路径变了"会被静默吞成
# 空注册，训练启动时才报 "reward func not found"，那时候排查成本已经高一个量级了。
_SWIFT_AVAILABLE = True
try:
    from swift.plugin import ORM, orms
except ModuleNotFoundError:
    try:
        from swift.rewards import ORM, orms           # 部分版本把它挪到了这里
    except ModuleNotFoundError:
        _SWIFT_AVAILABLE = False
        print("[grpo_reward] 未找到 ms-swift 的 ORM 注册接口，奖励函数没有注册。"
              "本地单测不受影响；训练环境里出现这条就是环境不对。", file=sys.stderr)

        class ORM:                                    # type: ignore[no-redef]
            """本地占位，让单元测试能在没装 swift 的机器上导入本模块。"""

        orms: dict = {}

# ---------------------------------------------------------------------------
# 重复惩罚的三组阈值：(片段长度, 出现次数阈值, 扣分)
# 短片段要重复更多次才算异常，长片段重复三次就已经很可疑了。
# 这三组是针对"生成到最大长度、末尾开始复读"这种典型退化设的，不是通用文本质量指标。
# ---------------------------------------------------------------------------
_REPEAT_RULES = ((8, 5, 0.35), (12, 4, 0.35), (20, 3, 0.25))
# 重复惩罚封顶。留一点余量，让"重复严重"和"格式失败"之间仍有区分度。
_REPEAT_CAP = 0.8
# 太短的文本本来就容易有重复片段，不判。
_REPEAT_MIN_CHARS = 80
# 超长惩罚：每超出 1000 字扣这么多，封顶 0.6
_LENGTH_PENALTY_PER_1K = 0.3
_LENGTH_PENALTY_CAP = 0.6

_online_call_count = 0


def _cfg():
    """取奖励段配置。"""
    return load_config().reward


def _extract_text(completion) -> str:
    """把 ms-swift 传来的 completion 统一成字符串。

    不同版本传的可能是 str、``[{'role','content'}]`` 或 dict，这里一次都兼容掉。

    :param completion: 框架传来的单条生成
    :return: 生成文本
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content", "")
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _expand_column(values, n: int, name: str) -> list:
    """把数据集的一列对齐到 completions 的长度。

    GRPO 每个 prompt 采 K 条，所以 completions 长度是 prompt 数 × K，而数据列还是 prompt 长度。
    按整除关系展开，每个值重复 K 次。

    不整除就报错，绝不静默截断——截断意味着某些候选会拿到别的题的答案池去比，
    奖励算得完全是错的，而训练照常跑，曲线还挺好看。

    :param values: 数据列
    :param n: completions 条数
    :param name: 列名，只用于报错
    :return: 长度为 n 的列表
    :raise KeyError: 列不存在
    :raise ValueError: 长度不匹配且不整除
    """
    if values is None:
        raise KeyError(f"GRPO 数据缺 {name} 列，拒绝继续。")
    if not isinstance(values, list):
        values = [values]
    if len(values) == n:
        return values
    if len(values) and n % len(values) == 0:
        repeat = n // len(values)
        return [v for v in values for _ in range(repeat)]
    raise ValueError(f"completions({n}) 与 {name}({len(values)}) 长度不匹配且不整除。")


def _parse_pool(value) -> list[str]:
    """把随行带的答案池解回列表。

    行里存的是 JSON 字符串，但框架在某些路径下会把它还原成 list，两种都接。

    :param value: `v1_answers_json` 列的值
    :return: 答案列表
    """
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x or "").strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x or "").strip()]
    except json.JSONDecodeError:                       # noqa: PERF203 - 不是 JSON 就当单条答案处理
        pass
    return [text]


@contextlib.contextmanager
def _online_judge_guard():
    """给在线判分调用串一把跨进程文件锁。

    GRPO 的奖励插件会被多个分布式 worker 各自导入一份。不加锁的话，八个 worker 同时往
    裁判服务打请求，几乎必然触发限流，然后八个一起退避、一起重试，把训练卡在那里。
    一把小文件锁让流量变成可预测的串行。

    锁只罩在线奖励这一处。离线评测有自己的并发池，不受影响。
    """
    cfg = load_config()
    lock_path = Path(cfg.paths["log_dir"]) / "grpo_judge.lock"
    try:
        import fcntl                                   # Linux 训练机有；Windows 上跑单测没有
    except ImportError:
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _judge_score(reference: str, think: str, k: int, required: bool) -> float | None:
    """在线判分：打 k 遍取平均。

    这里的 k 通常是 2，**不是**选样口径。它只在一组候选内部排序，不套 σ 标定表，
    所以不需要 16 遍。

    :param reference: 该题参考资料
    :param think: 被判的推理段
    :param k: 打几遍
    :param required: True 则全失败时抛异常停训；False 则返回 None 降级成"只看规则"
    :return: 平均分，或 None
    :raise RuntimeError: `required` 为 True 且 k 遍全失败
    """
    cached = budget.cache_get(reference, think, k)
    if cached is not None:
        return cached.get("clean_score")

    scores: list[float] = []
    last_error: Exception | None = None
    for _ in range(max(1, k)):
        try:
            with _online_judge_guard():
                scores.append(float(judge_clean_score(reference, think)["clean_score"]))
        except Exception as exc:                       # noqa: BLE001 - 单遍失败降级成少打一遍，全失败才由 required 决定
            last_error = exc
    if not scores:
        if required:
            raise RuntimeError(f"在线 GRPO 判分全部失败（k={k}）: {last_error!r}")
        return None

    result = {"clean_score": sum(scores) / len(scores), "n": len(scores),
              "sd": statistics.stdev(scores) if len(scores) >= 2 else 0.0}
    budget.cache_put(reference, think, k, result)
    return result["clean_score"]


def _repeat_penalty(text: str) -> float:
    """重复片段惩罚。

    专治两种退化：生成撞到最大长度后开始复读；以及模型发现某句话能拿分就一直念。
    做法是数各长度的字符片段最多重复了几次，超阈值就扣分。

    :param text: 推理段
    :return: 惩罚值，0 ~ :data:`_REPEAT_CAP`
    """
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < _REPEAT_MIN_CHARS:
        return 0.0
    penalty = 0.0
    for size, threshold, weight in _REPEAT_RULES:
        counts: dict[str, int] = {}
        for i in range(0, max(0, len(compact) - size + 1)):
            gram = compact[i:i + size]
            counts[gram] = counts.get(gram, 0) + 1
        if counts and max(counts.values()) >= threshold:
            penalty += weight
    return min(_REPEAT_CAP, penalty)


def _length_penalty(think: str, max_chars: int) -> float:
    """超长惩罚：超出上限之后按超出量线性扣，封顶 :data:`_LENGTH_PENALTY_CAP`。

    不做硬截断而做软惩罚，是因为长度和质量不是单调关系——有的题就是要多推两步。
    硬截断会把这类题一刀切掉。

    :param think: 推理段
    :param max_chars: 长度上限
    :return: 惩罚值
    """
    length = len((think or "").strip())
    if length <= max_chars:
        return 0.0
    return min(_LENGTH_PENALTY_CAP, (length - max_chars) / 1000.0 * _LENGTH_PENALTY_PER_1K)


def reward_one(text: str, user_prompt: str, pool_answers: list[str], *, use_judge: bool) -> float:
    """给一条候选打一个标量分。整个奖励逻辑就这一个函数。

    :param text: 候选的完整生成文本
    :param user_prompt: 该题题面
    :param pool_answers: 该题的基座认可池
    :param use_judge: 这一步要不要调裁判（warmup 阶段不调）
    :return: 标量奖励
    :raise ValueError: 该题没带答案池——在线训练不能没有硬门靶子
    """
    cfg = _cfg()

    # 1.【格式门】标签顺序必须严格是 <think>…</think>…<answer>…</answer>
    if bool(cfg["strict_tags"]):
        open_think = text.find("<think>")
        close_think = text.find("</think>")
        open_answer = text.find("<answer>",
                                close_think + len("</think>") if close_think != -1 else 0)
        close_answer = text.rfind("</answer>")
        if not (0 <= open_think < close_think < open_answer < close_answer):
            return float(cfg["format_fail"])

    parsed = parse_think_answer_diagnostic(text)
    think, answer = parsed["think"], parsed["answer"]
    if (not parsed["format_ok"]) or len(think.strip()) < int(cfg["think_min_chars"]):
        return float(cfg["format_fail"])

    if not pool_answers:
        raise ValueError("在线 GRPO 的这条数据没带答案池，硬门没有靶子，拒绝训练。")

    # 2.【答案门】漂出池就是负分，后面什么都不看了
    verdict = answer_in_v1_pool(answer, pool_answers)
    if not verdict["in_pool"]:
        return float(cfg["answer_out_of_pool"])
    if not verdict.get("comparable", True):
        # 离线评测为了和历史口径可比，会把"非空但没有可抽事实"的答案仍算进在池率。
        # 在线训练不能这么放：那等于告诉模型"少说具体数字、别提金额和日期，就能安全拿分"，
        # 恰好把 grounding 反向优化掉。
        return float(cfg["answer_not_comparable"])

    # 3.【规则门 + 裁判排序】只有前两道门都过了才可能调裁判
    rule = detect_rag_style(think)
    judge_score = None
    if use_judge and not rule["has_rag_style"]:
        online_cfg = load_config().train["grpo"]["online"]
        judge_score = _judge_score(
            extract_references(user_prompt), think,
            int(online_cfg["judge_k"]), bool(online_cfg["judge_required"]))
    # 裁判分是 0~10，归一到 0~1 再加权，让它和下面那些档位常数在同一个量级上
    normalized = 0.0 if judge_score is None else max(0.0, min(1.0, float(judge_score) / 10.0))
    penalty = _repeat_penalty(think) + _length_penalty(think, int(cfg["think_max_chars"]))

    if rule["has_rag_style"]:
        # 规则没过这一档，裁判分的权重明显更低。这样即使裁判给了满分，
        # 也不可能反超一个规则通过的候选——词典序在这里靠数值区间保证。
        return round(float(cfg["rule_fail_base"])
                     + float(cfg["rule_fail_judge_weight"]) * normalized - penalty, 4)
    return round(float(cfg["rule_pass_base"])
                 + float(cfg["rule_pass_judge_weight"]) * normalized - penalty, 4)


def _reward_batch(completions, kwargs, *, use_judge: bool) -> list[float]:
    """给一整批 completions 打分。

    :param completions: 框架传来的候选列表
    :param kwargs: 框架传来的数据列
    :param use_judge: 要不要调裁判
    :return: 与 completions 等长的奖励列表
    """
    n = len(completions)
    prompts = _expand_column(kwargs.get("user_prompt"), n, "user_prompt")
    pools_raw = kwargs.get("v1_answers_json")
    if pools_raw is None:
        pools_raw = kwargs.get("v1_answers")
    pools = _expand_column(pools_raw, n, "v1_answers_json")
    # strict=True：三者长度已由 _expand_column 对齐，再对不上就是框架行为变了，
    # 必须当场炸而不是按最短截断——截断意味着一部分候选拿别的题的池去比，奖励全错。
    return [float(reward_one(_extract_text(c), str(p or ""), _parse_pool(pool), use_judge=use_judge))
            for c, p, pool in zip(completions, prompts, pools, strict=True)]


class RuleWarmupReward(ORM):
    """预热奖励：只看格式、答案、规则，一次裁判都不调。

    它是**刻意短命的**。作用是在花钱调裁判之前，先把生成格式和答案池稳住——
    模型刚从偏好学习切到在线采样时，格式失败率会短暂抬头，这时候每条都去调裁判纯属浪费。

    跑久了会有反效果：只有规则这一档信号，模型会把表面规则刷满（该说的检索腔词一个不说），
    而换词复述照抄一点没改。所以配置里给它的步数只有在线阶段的三分之一。
    """

    def __call__(self, completions, **kwargs) -> list[float]:
        return _reward_batch(completions, kwargs, use_judge=False)


class OnlineReward(ORM):
    """在线奖励：完整的词典序硬门 + 裁判在安全区内排序。

    见模块开头对词典序的说明。
    """

    def __call__(self, completions, **kwargs) -> list[float]:
        global _online_call_count
        _online_call_count += 1
        return _reward_batch(completions, kwargs, use_judge=True)


# 注册名 = 训练脚本里 --reward_funcs 用的名字
orms["rule_warmup"] = RuleWarmupReward
orms["online"] = OnlineReward

# 装了 swift 却没注册进去是致命的：训练要跑到解析参数那一步才报错，而那时候
# 权重已经加载、卡已经占上了。启动即响。
assert (not _SWIFT_AVAILABLE) or orms.get("online") is OnlineReward, \
    "奖励函数没注册进 ms-swift 的 orms，核对当前 swift 版本的 ORM 注册接口。"
