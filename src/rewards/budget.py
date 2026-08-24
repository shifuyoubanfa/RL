"""裁判调用的围栏：记账 + 去重缓存 + 预算硬闸。

本模块在整条链路里的位置：旁路。挂在裁判客户端和打分函数旁边，不参与任何判定逻辑，
所以关掉它不影响实验结论。

三件事：

1. **记账**。每次调用把返回里的 token 用量累加进 `output_dir/judge_budget.json`，
   按标价折算成钱。监控直接读这个文件。
2. **无损去重缓存**。同一组 `(参考资料, think, k)` 的**聚合分**跨阶段只算一次。
   它不碰"打 k 遍取平均"这件事本身——降噪照旧，只是同一个估计量不重复估。
   基座原推理这条对照锚点在冷启动、拒绝采样、偏好对三个阶段里被反复评，缓存后只评一遍，
   统计上完全等价（选样判据用的是均值加标定表的 σ，不用这 k 遍的经验标准差）。
3. **预算硬闸**。累计花费超过围栏就抛异常，整步干净退出，由人决定要不要加预算续跑。
   这是**安全闸不是降质**：到顶了是停，不是偷偷把 k 调小接着跑。

只缓存 k ≥ `cache_min_k` 的选样分。k=2 的粗筛短命、候选各不相同，缓存了只会把文件撑爆
而几乎不命中。
"""

from __future__ import annotations

import atexit
import hashlib
import json
import threading
from pathlib import Path

from src.config import load_config

_lock = threading.Lock()
_meter = {"calls": 0, "in_tokens": 0, "out_tokens": 0, "yuan": 0.0, "cache_hits": 0}
_cache: dict[str, dict] = {}
_loaded = False
_warned = False

# 花到围栏的这个比例时告警一次。
_WARN_FRACTION = 0.8
# 每这么多次调用把账落一次盘。太密是无谓的磁盘写，太疏则监控页数字滞后。
_FLUSH_EVERY = 20


class BudgetExceeded(BaseException):
    """累计花费超过围栏。

    **刻意继承 BaseException 而不是 Exception。** 打分和改写这条路上到处是
    `except Exception` 的容错重试（一次判分失败就再试一次是对的）。如果这个异常继承
    Exception，它会被那些容错分支静默吞掉——围栏就成了摆设，钱照烧。
    继承 BaseException 让它穿过所有容错层，一路上抛到步骤进程退出。
    """


def _paths() -> tuple[Path, Path]:
    """返回 (记账文件, 缓存文件)。两个都落在 output_dir 根下，跨阶段共享。"""
    output_dir = Path(load_config().paths["output_dir"])
    return output_dir / "judge_budget.json", output_dir / "judge_score_cache.jsonl"


def _yuan(in_tokens: int, out_tokens: int) -> float:
    """按配置里的标价把 token 折成钱。"""
    cfg = load_config().judge
    return (in_tokens / 1e6 * float(cfg["price_in_per_million"])
            + out_tokens / 1e6 * float(cfg["price_out_per_million"]))


def _load() -> None:
    """把上一步留下的累计账和缓存读进来。步骤是顺序跑的独立进程，跨步累计才对得上。"""
    global _loaded
    if _loaded:
        return
    meter_file, cache_file = _paths()
    try:
        if meter_file.exists():
            saved = json.loads(meter_file.read_text(encoding="utf-8"))
            for key in ("calls", "in_tokens", "out_tokens", "cache_hits"):
                _meter[key] = int(saved.get(key, 0))
            _meter["yuan"] = float(saved.get("yuan", 0.0))
    except Exception:                                  # noqa: BLE001 - 账本读坏了从零记，绝不能因此挡住真实调用
        pass

    cfg = load_config().judge
    if cache_file.exists() and int(cfg["cache_min_k"]) > 0:
        with cache_file.open("r", encoding="utf-8") as f:
            for line in f:                             # 逐行容错：崩溃留下的半行只跳它自己
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    _cache[entry["key"]] = entry["val"]
                except Exception:                      # noqa: BLE001 - 坏行跳过，其余缓存照用
                    continue
    _loaded = True


def _flush_meter() -> None:
    """把账落盘。失败只吞掉，记账绝不能影响真实调用。"""
    meter_file, _ = _paths()
    tmp = meter_file.with_suffix(".json.tmp")
    try:
        meter_file.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(_meter, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(meter_file)
    except Exception:                                  # noqa: BLE001 - 落盘失败最多丢一段账，不值得中断流程
        pass


def record(usage: dict | None) -> None:
    """上报一次调用的 token 用量，顺便检查围栏。

    告警状态和超限判断都在锁内定下来，锁外才打印和抛出——否则多线程下会重复告警。

    :param usage: 服务端返回的 usage 字段
    :raise BudgetExceeded: 累计花费达到围栏
    """
    global _warned
    should_warn = over = False
    budget_yuan = float(load_config().judge["budget_yuan"])
    with _lock:
        _load()
        in_tokens = int((usage or {}).get("prompt_tokens", 0) or 0)
        out_tokens = int((usage or {}).get("completion_tokens", 0) or 0)
        _meter["calls"] += 1
        _meter["in_tokens"] += in_tokens
        _meter["out_tokens"] += out_tokens
        _meter["yuan"] = round(_meter["yuan"] + _yuan(in_tokens, out_tokens), 4)
        if _meter["calls"] % _FLUSH_EVERY == 0 or _meter["calls"] < 5:
            _flush_meter()
        current = _meter["yuan"]
        if budget_yuan > 0:
            if not _warned and current >= _WARN_FRACTION * budget_yuan:
                _warned = should_warn = True
            if current >= budget_yuan:
                over = True
                _flush_meter()                         # 抛之前确保最新累计已落盘，续跑基线才不丢

    if should_warn:
        print(f"[budget] 已花 ¥{current:.1f} / 围栏 ¥{budget_yuan:.0f}"
              f"（{current / budget_yuan:.0%}）", flush=True)
    if over:
        raise BudgetExceeded(
            f"裁判调用累计 ¥{current:.1f} 已达围栏 ¥{budget_yuan:.0f}。"
            f"加大 judge.budget_yuan 后续跑（缓存还在，不重烧）；设成 0 关闸。")


def _key(reference: str, think: str, k: int) -> str:
    """缓存键 = 版本串 + 参考资料 + think + k 的哈希。

    版本串来自配置里的 `cache_version`：判分提示词一改，旧缓存就该整体失效，
    不然新口径的分和旧口径的分会混在一起，选样门用的是哪一套完全说不清。
    """
    cfg = load_config().judge
    hasher = hashlib.sha1()
    for part in (str(cfg["cache_version"]), reference or "", think or "", str(k)):
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def cache_get(reference: str, think: str, k: int) -> dict | None:
    """查缓存。k 小于 `cache_min_k` 直接不查。

    :return: 命中的聚合分，没命中返回 None
    """
    cfg = load_config().judge
    if k < int(cfg["cache_min_k"]):
        return None
    with _lock:
        _load()
        hit = _cache.get(_key(reference, think, k))
        if hit is None:
            return None
        _meter["cache_hits"] += 1
        return dict(hit)


def cache_put(reference: str, think: str, k: int, result: dict) -> None:
    """写缓存。只收**打满 k 遍**的聚合分。

    打了一半就失败的退化均值不进缓存：它只被 √n 收噪却会被当成 √k 的锚点永久留下，
    选样门用 k 档的 σ 去判就会偏松，毒化下游整批样本。

    :param reference: 参考资料
    :param think: 被打分的推理段
    :param k: 打了几遍
    :param result: 聚合结果，须含 `clean_score` 和 `n`
    """
    cfg = load_config().judge
    if k < int(cfg["cache_min_k"]):
        return
    if not result or result.get("n", 0) < k or result.get("clean_score") is None:
        return

    key = _key(reference, think, k)
    _, cache_file = _paths()
    with _lock:
        if key in _cache:
            return
        _cache[key] = dict(result)
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "val": result}, ensure_ascii=False) + "\n")
        except Exception:                              # noqa: BLE001 - 缓存写失败只是下次重算，不影响正确性
            pass


def snapshot() -> dict:
    """当前累计用量，给监控和收尾日志用。

    :return: ``{calls, in_tokens, out_tokens, yuan, cache_hits, budget_yuan, pct}``
    """
    budget_yuan = float(load_config().judge["budget_yuan"])
    with _lock:
        _load()
        snap = dict(_meter)
    snap["budget_yuan"] = budget_yuan
    snap["pct"] = (snap["yuan"] / budget_yuan) if budget_yuan > 0 else 0.0
    return snap


def flush() -> None:
    """手动落盘。"""
    with _lock:
        _flush_meter()


# 每个步骤都是独立子进程，正常退出时没人调 flush，尾账（最多 _FLUSH_EVERY-1 次）就丢了。
# 并发执行器用的是 with-ThreadPoolExecutor，返回前线程已全部 join，atexit 时不会有人持锁。
atexit.register(flush)
