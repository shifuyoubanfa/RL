"""第 1 步：让基座对全部题目重新生成一遍 think 和 answer。

本模块在整条链路里的位置：最上游。它把"一批题面"变成"整条链路的原料"。

输入：`paths.raw_corpus`，每行 ``{query, user_prompt}``。`user_prompt` 是完整题面，
里面含【参考问答对】和【问题】两段。

输出：`00_base_outputs.jsonl`，每行 ``{query, user_prompt, raw, reasoning, answer}``。

- ``answer`` 是**答案金标准**。后面所有阶段的训练样本，answer 段都拼它（answer-lock）。
- ``reasoning`` 是基座原推理，检索腔满满的那一版。它有两个用途：偏好对里当 rejected，
  选样时当"脏"对照。

**为什么要重产，而不是直接用语料里已有的答案。** 因为链路要保证的是"答案不漂离**这个基座**"。
拿别处来的答案当锚，模型只要和这个基座本来的行为不一致就会被判漂——那测的是基座和别人的差异，
不是训练带来的变化。

生成走贪心（温度 0），让这份金标准可复现。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from src.config import Config
from src.data import paths
from src.data.parsing import parse_think_answer
from src.data.prompts import RAG_SYSTEM_PROMPT
from src.models import vllm_client
from src.utils.concurrency import map_concurrent
from src.utils.log import get_logger

log = get_logger("build_base_outputs")


def _load_done_queries(path: Path) -> set[str]:
    """读已完成的 query，用来续跑跳过。

    逐行容错：这个文件是边跑边追加的，被 kill 时最后一行可能只写了一半。

    :param path: 输出文件路径
    :return: 已完成的 query 集合
    """
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["query"])
            except Exception:                          # noqa: BLE001 - 半行跳过，其余进度照用
                continue
    return done


def run(cfg: Config, *, limit: int = 0) -> Path:
    """跑第 1 步。

    :param cfg: 配置
    :param limit: 只处理前 N 条，0 = 全量。联调用
    :return: 输出文件路径
    :raise SystemExit: 找不到原始语料
    """
    source = Path(cfg.paths["raw_corpus"])
    out_path = paths.base_outputs(cfg)
    if not source.exists():
        raise SystemExit(
            f"缺少原始语料 {source}。每行需要 {{query, user_prompt}} 两个字段，"
            f"格式见 examples/sample_data.jsonl。")

    vllm_client.wait_ready()

    with source.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if limit:
        records = records[:limit]

    done = _load_done_queries(out_path)
    todo = [r for r in records if r.get("query") not in done]
    log.info("基座重产：待处理 %d / %d（已完成 %d）", len(todo), len(records), len(done))

    data_cfg = cfg.data
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 每完成一条立刻追加写并 flush。不这么做，被 kill 时 _load_done_queries 读不到任何东西，
    # 续跑等于从头重跑——这一步是整条链路里最慢的几步之一。
    write_lock = threading.Lock()
    sink = out_path.open("a", encoding="utf-8")

    def _generate(record: dict) -> dict:
        user_prompt = record.get("user_prompt") or ""
        raw = vllm_client.generate_one(
            user_prompt, system=RAG_SYSTEM_PROMPT,
            temperature=float(data_cfg["build_temperature"]),
            top_p=float(data_cfg["build_top_p"]),
            max_tokens=int(data_cfg["build_max_tokens"]))
        think, answer = parse_think_answer(raw)
        row = {"query": record.get("query"), "user_prompt": user_prompt,
               "raw": raw, "reasoning": think, "answer": answer}
        with write_lock:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
        return row

    try:
        results = map_concurrent(todo, _generate, workers=int(cfg.vllm["call_workers"]), desc="基座重产")
    finally:
        sink.close()

    empties = sum(1 for r in results if not (r["answer"] or "").strip())
    log.info("完成：新增 %d 条 -> %s（其中空答案 %d 条，需关注）", len(results), out_path, empties)
    return out_path
