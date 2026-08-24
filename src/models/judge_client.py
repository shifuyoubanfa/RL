"""裁判模型客户端：OpenAI 兼容的 chat completions，带退避重试和用量上报。

本模块在整条链路里的位置：所有需要大模型判断的地方（冷启动改写、选样打分、在线奖励、
离线评测）唯一的出口。

三条红线：

1. **key 只从环境变量读**，代码里没有任何兜底常量。本仓库是公开的。
2. **判分参数显式锁死**（温度 0），不依赖模型名分支。判分要可复现，选样和评测才可比。
3. **重试前换新连接**。这是一个实测教训：服务端抖动之后，另起一个新进程两秒就能打上分，
   而长跑的训练进程却一直超时——旧连接池里的 keep-alive socket 在抖动里坏了，
   复用它只会一直等下去。重建连接的代价对判分这种网络瓶颈任务可以忽略。
"""

from __future__ import annotations

import contextlib
import random
import time

import requests

from src.config import judge_api_key, load_config
from src.rewards import budget
from src.utils.log import get_logger

log = get_logger("judge_client")

_session = requests.Session()

# 退避封顶秒数。再长就不如让整步失败、由人来决定要不要续跑。
_BACKOFF_CAP = 30.0
# 退避随机抖动上限秒数。多个 worker 同时被 429 时错开重试，避免整齐撞上去又一起被拒。
_JITTER = 3.0


def _reset_session() -> None:
    """丢掉旧连接池，换一个新的。见模块开头第 3 条。"""
    global _session
    # 关旧连接失败不影响新连接：这里要的只是"别再用那个坏 socket"，旧的没关干净由 GC 收
    with contextlib.suppress(Exception):
        _session.close()
    _session = requests.Session()


def chat(messages: list[dict], *, model: str | None = None, temperature: float | None = None,
         top_p: float | None = None, max_tokens: int = 4096,
         timeout: int | None = None, retries: int | None = None) -> str:
    """调一次 chat completions，返回 content 字符串。

    对 429 和 5xx 和超时做指数退避重试；对 4xx（429 除外）立即抛出——那是模型名写错、
    鉴权失败或参数非法，重试多少次都是一样的结果，早点报出来比空等强。

    用量上报（:func:`src.rewards.budget.record`）放在重试循环之外。放里面的话，
    预算超限抛出的异常会被上面那个 `except Exception` 当成一次调用失败吃掉，围栏就形同虚设。

    :param messages: OpenAI 格式的消息列表
    :param model: 覆盖配置里的模型名
    :param temperature: 覆盖判分温度
    :param top_p: 覆盖 top_p
    :param max_tokens: 最大生成长度
    :param timeout: 单次请求读超时（秒）
    :param retries: 重试次数
    :return: 模型回复的 content
    :raise RuntimeError: 重试用尽仍失败，或遇到永久性 4xx
    :raise src.rewards.budget.BudgetExceeded: 累计花费超过围栏
    """
    cfg = load_config().judge
    retries = int(retries if retries is not None else cfg["retries"])
    url = str(cfg["base_url"]).rstrip("/") + "/chat/completions"
    payload = {
        "model": model or cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"] if temperature is None else temperature,
        "top_p": cfg["top_p"] if top_p is None else top_p,
        "max_tokens": max_tokens,
        "stream": False,
        # 该裁判是 thinking 模型，不关思考的话正文全落 reasoning_content、content 是空的
        "enable_thinking": bool(cfg["enable_thinking"]),
    }
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + judge_api_key(load_config())}

    last_error: Exception | None = None
    content = usage = None
    for attempt in range(retries):
        if attempt > 0:
            _reset_session()
        try:
            response = _session.post(url, json=payload, headers=headers,
                                     timeout=timeout or cfg["timeout"])
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage")
            break
        except requests.HTTPError as exc:
            code = getattr(exc.response, "status_code", 0)
            body = getattr(exc.response, "text", "")
            if 400 <= code < 500 and code != 429:
                raise RuntimeError(f"裁判服务永久性错误 {code}（模型名/鉴权/参数？）: {body[:300]}") from exc
            last_error = exc
            log.warning("裁判服务第 %d 次失败(%s): %s", attempt + 1, code, body[:200])
            time.sleep(min(_BACKOFF_CAP, 2 ** attempt) + random.uniform(0, _JITTER))
        except Exception as exc:                       # noqa: BLE001 - 网络层任何异常都退避重试，用尽后统一抛
            last_error = exc
            log.warning("裁判服务第 %d 次失败: %r", attempt + 1, exc)
            time.sleep(min(_BACKOFF_CAP, 2 ** attempt) + random.uniform(0, _JITTER))
    else:
        raise RuntimeError(f"裁判服务调用失败（重试 {retries} 次）: {last_error}")

    budget.record(usage)
    return content


def smoke() -> str:
    """最小连通冒烟：让裁判回一句话。

    在占用 GPU 之前先跑它。等训练起来了才发现 key 过期，代价是整轮排队时间。

    :return: 模型的回复文本
    """
    return chat([{"role": "user", "content": "回复两个字：在么"}], max_tokens=16)
