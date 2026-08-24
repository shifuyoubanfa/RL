"""JSONL 读写与"攒够就停"的分批执行器。

本模块在整条链路里的位置：所有数据构建步骤的地基。链路里每一步的输入输出都是 jsonl，
每一步都可能跑几个小时、中途被 kill，所以读写必须原子、进度必须可续。

三件事：

1. **原子落盘**：先写 `.tmp` 再 `replace`。进程被 kill 只会留下一个 `.tmp`，
   不会留下一个"看起来完整、其实截断"的正式文件——后者会被续跑当成已完成直接跳过。
2. **qid**：同一道题在链路里跨五六个文件流转，靠 `sha1(query)[:12]` 当主键对齐。
3. **`gather_until`**：分批跑、边跑边落盘、攒够目标就停。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Callable, Iterable, Sequence

from src.utils.log import get_logger

log = get_logger("jsonl_io")


def qid_of(query: str) -> str:
    """题目主键 = query 的 sha1 前 12 位。

    用内容哈希而不是行号：链路中途会去重、洗牌、早停，行号根本对不上；
    截 12 位是因为全量题库在千条量级，碰撞概率可以忽略，日志里也短得能看。

    :param query: 题面原文
    :return: 12 位十六进制字符串
    """
    return hashlib.sha1((query or "").encode("utf-8")).hexdigest()[:12]


def read_jsonl(path: str | Path) -> list[dict]:
    """读一个 jsonl，文件不存在返回空列表。

    不存在返回空而不是报错：链路里很多步会先看"上游产物在不在"来决定跳不跳过，
    让调用方用 `if not read_jsonl(p)` 判断比到处 try 干净。

    :param path: jsonl 路径
    :return: 每行解析出的 dict 组成的列表
    :raise json.JSONDecodeError: 文件里有坏行（正式产物有坏行就是真出事了，不容错）
    """
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """原子写 jsonl：先写同目录 `.tmp`，写完 `replace` 成正式文件。

    同目录是必须的——跨文件系统 `replace` 不是原子操作。

    :param path: 目标路径，父目录不存在会自动建
    :param rows: 要写的记录
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(p)


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """追加写 jsonl，每批 flush 一次。

    进度文件专用。它跟 :func:`write_jsonl` 的取舍正好相反：进度文件容忍最后一行写半截
    （读的时候逐行容错跳过），换来的是崩溃时前面已完成的批次一条都不丢。

    :param path: 目标路径
    :param rows: 本批结果
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_progress(path: str | Path, key: str = "qid") -> dict[str, dict]:
    """读进度文件，按 key 去重返回 {键值: 记录}。

    逐行容错：崩溃时留下的半行只跳过它自己，不能因为最后一行坏了就把前面几千条一起丢掉。

    :param path: 进度文件路径
    :param key: 用哪个字段当主键
    :return: 主键到记录的映射；文件不存在返回空 dict
    """
    done: dict[str, dict] = {}
    p = Path(path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:              # noqa: PERF203 - 半行只跳它自己，其余进度照用
                continue
            if row.get(key) is not None:
                done[row[key]] = row
    return done


def index_by(rows: Sequence[dict], key: str = "qid") -> dict[str, dict]:
    """把记录列表转成 {主键: 记录}，缺主键的行丢掉。

    :param rows: 记录列表
    :param key: 主键字段名
    :return: 主键到记录的映射
    """
    return {r[key]: r for r in rows if r.get(key)}


def nonempty(path: str | Path) -> bool:
    """文件存在且大于 0 字节。

    专门区分"没产出"和"产出了但是空的"：选样门一条都没选中是合法结果，
    会写出一个 0 字节文件；下游据此跳过该阶段，而不是当成"上一步没跑"再跑一遍。

    :param path: 文件路径
    :return: 存在且非空为 True
    """
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def gather_until(
    items: Sequence[dict],
    fn: Callable[[dict], dict],
    *,
    enough: Callable[[list], bool],
    chunk: int,
    workers: int,
    desc: str,
    progress_path: str | Path | None = None,
    key: str = "qid",
    seed: int = 42,
) -> list[dict]:
    """先洗牌，再分批并发跑 `fn`，每批完检查一次"够没够"，够了就停。

    这个函数只决定**处理多少道题**。每道题怎么处理（采几条、判几遍、过几道门）全在 `fn`
    里，本函数一概不碰——这条边界是刻意的：早停如果顺手把 k 值也调小，省下来的钱会以
    "选样判据偏松、噪声样本混进训练集"的形式在下游还回去。

    先洗牌再分批，早停拿到的是**随机代表子集**，不是"题库前 N 条"。固定 seed 保证
    续跑接得上同一个顺序。

    落盘时机是"整批返回后"。中途被 kill，正在跑的那一批（最多 `chunk` 道题）不进进度文件，
    续跑会重做一遍——重做的只有便宜的粗筛，贵的选样分已经在判分缓存里。想缩小这个窗口
    就把 `chunk` 调小。

    :param items: 待处理项，每项须带 `key` 字段
    :param fn: 单项处理函数，返回的 dict 也须带 `key` 字段
    :param enough: 接收"至今全部结果"，返回够没够
    :param chunk: 每批多少项
    :param workers: 批内并发数
    :param desc: 日志前缀
    :param progress_path: 进度文件；给了就边攒边落盘并支持续跑
    :param key: 主键字段名
    :param seed: 洗牌种子
    :return: 已处理项的结果列表（可能少于 items，因为早停）
    """
    import random

    from src.utils.concurrency import map_concurrent

    # 1.【续跑】把上次已经处理过的读回来，够了就整步跳过
    done = read_progress(progress_path, key) if progress_path else {}
    results = list(done.values())
    if results and enough(results):
        log.info("%s 续跑：已有 %d 条达标，整步跳过", desc, len(results))
        return results

    # 2.【洗牌】只对没做过的洗牌。固定 seed → 续跑顺序一致
    pool = [it for it in items if it.get(key) not in done]
    random.Random(seed).shuffle(pool)
    if done:
        log.info("%s 续跑：已攒 %d 条，剩 %d 题待处理", desc, len(results), len(pool))

    # 3.【分批】跑一批、落一批、判一次够没够
    for start in range(0, len(pool), max(1, chunk)):
        batch = map_concurrent(pool[start:start + chunk], fn, workers=workers, desc=desc)
        if progress_path:
            append_jsonl(progress_path, batch)
        results.extend(batch)
        if enough(results):
            log.info("%s 早停：攒够目标（本轮处理 %d/%d 题，共 %d 条）",
                     desc, start + len(batch), len(pool), len(results))
            return results
    log.info("%s 跑满（处理全部 %d 题，共 %d 条）", desc, len(pool), len(results))
    return results
