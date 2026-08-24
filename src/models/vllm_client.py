"""推理服务客户端：OpenAI 兼容 HTTP，调本机 vLLM。

本模块在整条链路里的位置：所有"要让模型生成点什么"的地方——重产基座输出、建答案池、
拒绝采样自采、偏好对 rollout、评测推理，全走它。

**为什么走 HTTP 而不是在进程里加载权重。** 一个 32B 模型加载一次要几分钟、占几十 G 显存。
数据构建这一步要对上千道题各生成若干条，如果每个步骤脚本自己加载一遍，光加载时间就
超过生成时间。把它 serve 成一个常驻服务，所有步骤发 HTTP 请求，加载只付一次，
而且 vLLM 的连续批处理能把并发请求合并成大 batch，吞吐比一条条来高一个量级。

**代价**：训练和推理抢同一批卡。所以编排器在每次训练前必须先把这个服务停掉，
见 :mod:`src.training.pipeline` 里的 `stop_vllm`。
"""

from __future__ import annotations

import time

import requests

from src.config import load_config
from src.data.prompts import RAG_SYSTEM_PROMPT
from src.utils.log import get_logger

log = get_logger("vllm_client")

_session = requests.Session()

# 健康检查的超时。服务没起来时要快速返回 False，不能在这卡住。
_HEALTH_TIMEOUT = 10


def health() -> bool:
    """服务在不在。查 `/models` 返回 200 即算就绪。

    :return: 就绪为 True
    """
    cfg = load_config().vllm
    try:
        response = _session.get(str(cfg["base_url"]).rstrip("/") + "/models", timeout=_HEALTH_TIMEOUT)
        return response.status_code == 200
    except Exception:                                  # noqa: BLE001 - 连不上就是没就绪，不需要区分原因
        return False


def wait_ready(max_wait: int = 1800, interval: int = 10) -> None:
    """阻塞等服务起好。

    默认等半小时：32B 权重要读几十 G，加上张量并行初始化，冷启动几分钟是常态，
    磁盘慢的机器更久。等不到就报错，不要静默往下走——后面每一条请求都会失败，
    错误信息还全是"连接被拒绝"，查起来绕远路。

    :param max_wait: 最多等多少秒
    :param interval: 每隔多少秒查一次
    :raise RuntimeError: 超时仍未就绪
    """
    cfg = load_config().vllm
    started = time.time()
    while time.time() - started < max_wait:
        if health():
            log.info("推理服务就绪 (%s)", cfg["base_url"])
            return
        log.info("等待推理服务起来... (%ds)", int(time.time() - started))
        time.sleep(interval)
    raise RuntimeError(f"推理服务 {max_wait}s 内未就绪: {cfg['base_url']}")


def chat(messages: list[dict], *, model: str | None = None, n: int = 1,
         temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 1536,
         timeout: int | None = None, retries: int = 4) -> list[str]:
    """调一次 chat completions，返回 n 条 content。

    `n > 1` 时一次请求返回 n 个候选。这比发 n 次请求省：prompt 只需前向一次，
    KV cache 在 n 个候选之间共享。rollout 采样全靠这个。

    :param messages: OpenAI 格式消息列表
    :param model: 请求哪个模型名，默认取配置里的 `served_model`
    :param n: 采几条候选
    :param temperature: 采样温度，0 即贪心
    :param top_p: 核采样阈值
    :param max_tokens: 最大生成长度
    :param timeout: 单次读超时（秒）
    :param retries: 重试次数，指数退避
    :return: n 条生成文本
    :raise RuntimeError: 重试用尽仍失败
    """
    cfg = load_config().vllm
    url = str(cfg["base_url"]).rstrip("/") + "/chat/completions"
    payload = {
        "model": model or cfg["served_model"], "messages": messages, "n": n,
        "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "stream": False,
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = _session.post(url, json=payload, timeout=timeout or cfg["timeout"])
            response.raise_for_status()
            return [choice["message"]["content"] for choice in response.json()["choices"]]
        except Exception as exc:                       # noqa: BLE001 - 本机服务，任何失败都值得退避重试
            last_error = exc
            log.warning("推理服务第 %d 次失败: %r", attempt + 1, exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"推理服务调用失败（重试 {retries} 次）: {last_error}")


def generate_one(user_prompt: str, *, system: str | None = None, **kwargs) -> str:
    """单条生成。数据构建和评测用，默认贪心。

    :param user_prompt: 题面
    :param system: system 提示，不给就用 RAG 腔（只有重产基座输出会走这个默认值）
    :return: 一条生成文本
    """
    messages = [{"role": "system", "content": system or RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
    return chat(messages, n=1, **kwargs)[0]


def generate_k(user_prompt: str, *, k: int, system: str | None = None,
               temperature: float | None = None, top_p: float | None = None, **kwargs) -> list[str]:
    """单题采 K 条候选。rollout 用。

    温度默认取配置里的 `rollout_temperature`，不是 0——贪心采出来的 K 条会几乎一模一样，
    筛选就没有可筛的空间了。

    :param user_prompt: 题面
    :param k: 采几条
    :param system: system 提示
    :param temperature: 覆盖采样温度
    :param top_p: 覆盖核采样阈值
    :return: k 条生成文本
    """
    data_cfg = load_config().data
    messages = [{"role": "system", "content": system or RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
    return chat(messages, n=k,
                temperature=float(data_cfg["rollout_temperature"]) if temperature is None else temperature,
                top_p=float(data_cfg["rollout_top_p"]) if top_p is None else top_p,
                **kwargs)
