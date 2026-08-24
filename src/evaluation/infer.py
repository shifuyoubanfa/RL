"""评测第一步：在冻结验收集上跑一遍推理，产出 think 和 answer。

本模块在整条链路里的位置：评测的前半段。它只负责生成，不打分——生成要占 GPU，打分只要网络，
分开之后可以生成完就把推理服务停掉，把卡让出来。

**评测一律贪心（温度 0）。** 采样会让同一个模型两次评测的分数不一样，那样阶段之间的
差值就分不清是训练带来的还是这次采样抖出来的。

**system 提示由模型名决定**（见 :func:`src.data.prompts.system_for`）。这是个语义开关：
基座要用它原生的 RAG 腔还原真实行为；训练后的模型要用中性提示，和训练数据对齐。
搞反了分数会全错，而且错得看不出来——模型照样能生成，只是腔调被提示词拉回去了。

解析用诊断版（:func:`src.data.parsing.parse_think_answer_diagnostic`），
生成被截断在 think 里的输出会被如实标成格式失败，绝不让容错兜底把残缺推理伪装成答案。
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.data.jsonl_io import read_jsonl, write_jsonl
from src.data.parsing import parse_think_answer_diagnostic
from src.data.prompts import system_for
from src.models import vllm_client
from src.utils.concurrency import map_concurrent
from src.utils.log import get_logger

log = get_logger("eval_infer")


def run(cfg: Config, *, model_name: str, eval_file: Path, out_path: Path) -> Path:
    """在验收集上跑推理。

    :param cfg: 配置
    :param model_name: 推理服务里该模型的名字
    :param eval_file: 冻结验收集路径
    :param out_path: 输出路径
    :return: 输出路径
    """
    vllm_client.wait_ready()
    system_prompt = system_for(model_name, str(cfg.vllm["served_model"]))
    records = read_jsonl(eval_file)
    log.info("评测推理：%d 条（model=%s, system=%s）", len(records), model_name,
             "RAG腔" if system_prompt.startswith("你是一个乐于助人") else "中性")

    def _infer(record: dict) -> dict:
        user_prompt = record.get("user_prompt") or ""
        raw = vllm_client.chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            model=model_name, n=1, temperature=0.0, top_p=1.0,
            max_tokens=int(cfg.data["build_max_tokens"]))[0]
        parsed = parse_think_answer_diagnostic(raw)
        return {"qid": record.get("qid"), "query": record.get("query"),
                "user_prompt": user_prompt, "gold_answer": record.get("answer", ""),
                "gen_text": raw, "think": parsed["think"], "answer": parsed["answer"],
                "format_ok": parsed["format_ok"], "format_reason": parsed["format_reason"]}

    results = map_concurrent(records, _infer, workers=int(cfg.vllm["call_workers"]),
                             desc=f"eval:{model_name}")
    write_jsonl(out_path, results)
    empties = sum(1 for r in results if not (r["answer"] or "").strip())
    log.info("完成：%d 条 -> %s（空答案 %d）", len(results), out_path, empties)
    return out_path
