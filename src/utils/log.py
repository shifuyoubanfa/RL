"""统一日志：同时写控制台和文件，级别各自可调。

本模块在整条链路里的位置：横切。每个步骤脚本、每个客户端都从这里取 logger。

统一在一处配置是为了两件事：所有子进程的日志能汇到同一个文件里连起来看；
以及"控制台只看进度、文件里留全量"这种分级只需要改一个地方。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# 已经配置过 handler 的 logger 名。重复配置会让同一条日志打印多遍。
_CONFIGURED: set[str] = set()


def _level(env_name: str, default: str) -> int:
    """从环境变量读日志级别，读不到或写错就用默认。

    :param env_name: 环境变量名
    :param default: 默认级别名，如 "INFO"
    :return: logging 模块的整数级别
    """
    raw = os.environ.get(env_name, default).upper()
    return getattr(logging, raw, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """取一个已配好 handler 的 logger。

    文件路径取 `RL_LOG_FILE`，没设就落到 `RL_LOG_DIR/pipeline.log`，再没设就是 `./runs/logs`。
    这里刻意不去 import 配置模块——日志要能在配置加载失败时也把错打出来。

    :param name: logger 名，一般用模块名
    :return: 配好控制台 + 文件 handler 的 logger
    """
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    log_dir = os.environ.get("RL_LOG_DIR", "./runs/logs")
    log_file = os.environ.get("RL_LOG_FILE", str(Path(log_dir) / "pipeline.log"))
    file_level = _level("RL_FILE_LOG_LEVEL", "INFO")
    console_level = _level("RL_CONSOLE_LOG_LEVEL", "INFO")

    logger.setLevel(min(file_level, console_level))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s", "%Y-%m-%d %H:%M:%S")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(console_level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # 不向 root 传播：否则 root 上如果也挂了 handler，每条日志会打两遍
    logger.propagate = False
    _CONFIGURED.add(name)
    return logger
