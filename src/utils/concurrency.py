"""并发执行器：把一批任务丢进线程池，保持输入顺序返回。

本模块在整条链路里的位置：横切。所有"对 N 道题各调一次外部服务"的步骤都走它。

用线程而不是进程：这些任务全是网络等待（推理服务、裁判服务），GIL 在等 socket 时是放开的，
线程池既够用又不用付进程间序列化的代价。
"""

from __future__ import annotations

import concurrent.futures as futures
import time
from typing import Any
from collections.abc import Callable, Sequence

from src.utils.log import get_logger

log = get_logger("concurrency")

# 每完成多少条打一行进度。太密会把训练日志淹掉，太疏又看不出卡没卡住。
_PROGRESS_EVERY = 50


def map_concurrent(items: Sequence[Any], fn: Callable[[Any], Any], *,
                   workers: int = 8, desc: str = "") -> list:
    """并发跑 `fn(item)`，结果按输入顺序返回。

    顺序必须保持：下游经常拿结果列表和输入列表 `zip` 起来用，乱序会静默错配。
    `as_completed` 拿到的是完成顺序，所以用下标写回预分配的列表，而不是 append。

    `fn` 里抛出的异常会在 `fut.result()` 处原样上抛，不在这里吞。单条失败该不该容错，
    是调用方的业务判断——比如采样失败可以跳过这道题，预算超限就必须整步停。

    :param items: 待处理项
    :param fn: 单项处理函数
    :param workers: 线程数
    :param desc: 日志前缀
    :return: 与 items 等长、同序的结果列表
    :raise Exception: `fn` 抛出的任何异常
    """
    results: list = [None] * len(items)
    if not items:
        return results

    done = 0
    started = time.time()
    with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        index_of = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in futures.as_completed(index_of):
            results[index_of[fut]] = fut.result()
            done += 1
            if done % _PROGRESS_EVERY == 0 or done == len(items):
                rate = done / max(time.time() - started, 1e-3)
                log.info("[%s] %d/%d  %.1f it/s", desc, done, len(items), rate)
    return results
