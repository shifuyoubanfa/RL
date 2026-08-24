"""第 4 步：冷启动 SFT 数据 —— 让裁判模型从头写一段自然推理，再过三道门。

本模块在整条链路里的位置：四个训练阶段的第一个。它要解决的是"模型压根不知道自然推理长什么样"
这个冷启动问题：直接上强化学习，采出来的 K 条候选全是检索腔，组内没有好样本可选，
优势估计全是噪声，学不动。

**三道门，从便宜到贵排列。**

1. **规则门**：改写稿里还带检索腔就直接丢。确定性判断，零成本。
2. **事实闸**：改写稿的结论极性不许跑到基座认可池外面去。也是确定性判断，
   放在裁判打分前面，省下一批必然会被丢掉的样本的打分费用。
3. **σ-可分门**：先打 `k_screen` 遍粗筛，均值没比基座原推理高就弃；过了再各打 `k_select` 遍，
   用标定表判两条误差带是否完全错开。

排列顺序不是随手定的：三道门里最贵的是第三道（每条要打十几遍分），
所以前两道确定性的门必须排在它前面，把注定过不了的样本先筛掉。

**事实闸为什么只拦极性，不拦数字。** 一个更严的做法是：改写稿里出现的任何数字，
只要不是从池子里逐字抄来的，就判臆造。这条路会系统性误杀最像人的那一批样本——
"9万还差1万到10万这条线"里的"1万"是算出来的，池子里没有，但它恰恰是"代入数字一步步推"的证据。
把这类样本全杀掉，剩下的就只有干巴巴罗列的。所以数字的合法算术派生放过，
只拦"推向池子里没有的结论极性"，也就是推出了相反或更宽更严的结论。
最终答案由 answer-lock 保着，结论数字由锚点锁在输入里，think 只需要不和池子矛盾。
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.data import paths
from src.data.build_answer_pool import load_pool_index
from src.data.jsonl_io import gather_until, read_jsonl, write_jsonl
from src.data.parsing import extract_references
from src.data.prompts import REWRITE_SYSTEM, REWRITE_TEMPLATE, build_pool_anchor
from src.data.schema import sft_row
from src.models import judge_client
from src.rewards.calibration import confident_cleaner
from src.rewards.judge import cleaner_scores
from src.rewards.rules import detect_rag_style, extract_facts
from src.utils.log import get_logger

log = get_logger("build_sft_data")

# 池子抽不出锚点时，回退成截断的原答案 prose。截这么长是为了不把改写输入撑爆，
# 同时留下足够的结论信息。
_ANCHOR_FALLBACK_CHARS = 1500
# 喂给改写模型的参考资料上限。比判分那边宽，因为改写要看全料才推得出来。
_REWRITE_REFERENCE_CHARS = 6000
_REWRITE_QUERY_CHARS = 500
# 训练桶少于这个数就告警：样本太少，SFT 学不到稳定的写法。
_MIN_USABLE_SAMPLES = 200

_pool_index: dict[str, dict] = {}


def _facts_ok(natural: str, record: dict, pool_answers: list[str]) -> bool:
    """事实闸：改写稿的结论极性不许跑到基座认可池外面。

    抽取口径和评测那道答案门完全一致（都走 :func:`src.rewards.rules.extract_facts`）。
    同口径是必须的：两边各用一套抽取，改写稿会在评测里凭空掉分，而且查不出原因。

    :param natural: 改写出来的推理段
    :param record: 该题的训练集记录，含基座 answer 和原 think
    :param pool_answers: 基座认可池里的答案
    :return: 没冲突为 True
    """
    pool_polarity: set[str] = set()
    for text in (record.get("answer"), record.get("reasoning"), *(pool_answers or [])):
        pool_polarity |= extract_facts(text or "")["polarity"]
    if not pool_polarity:
        return True                                    # 池子里一个极性词都没抽到，无从判，放过给后面两道门
    return not (extract_facts(natural)["polarity"] - pool_polarity)


def _process(cfg: Config, record: dict) -> dict:
    """处理一道题：改写 → 规则门 → 事实闸 → σ-可分门。

    :param cfg: 配置
    :param record: 训练集里的一行
    :return: 处理结果，含各门是否通过和两个选样分
    """
    gates = cfg.gates
    data_cfg = cfg.data
    references = extract_references(record.get("user_prompt") or "")
    base_think = (record.get("reasoning") or "").strip()
    pool_answers = (_pool_index.get(record.get("qid")) or {}).get("pool_answers") or []
    anchor = build_pool_anchor(pool_answers) or (record.get("answer") or "").strip()[:_ANCHOR_FALLBACK_CHARS]

    out = {"qid": record.get("qid"), "query": record.get("query"),
           "user_prompt": record.get("user_prompt"), "answer": record.get("answer"),
           "natural": None, "rule_ok": False, "facts_ok": False,
           "score_clean": None, "score_base": None, "selected": False}

    # 1.【改写】不喂基座原 think，只给问题 + 依据 + 锚点，让它从头自己推
    try:
        natural = judge_client.chat(
            [{"role": "system", "content": REWRITE_SYSTEM},
             {"role": "user", "content": REWRITE_TEMPLATE.format(
                 query=(record.get("query") or "")[:_REWRITE_QUERY_CHARS],
                 reference=references[:_REWRITE_REFERENCE_CHARS],
                 anchor=anchor)}],
            temperature=float(data_cfg["rewrite_temperature"]),
            top_p=float(data_cfg["rewrite_top_p"]),
            max_tokens=int(data_cfg["rewrite_max_tokens"])).strip()
    except Exception as exc:                           # noqa: BLE001 - 单题改写失败降级成"这题没样本"，漏斗日志里看得到
        log.warning("改写失败 qid=%s: %r", record.get("qid"), exc)
        return out
    out["natural"] = natural

    # 2.【规则门】改写稿自己还带检索腔，直接丢
    out["rule_ok"] = not detect_rag_style(natural)["has_rag_style"]
    if not out["rule_ok"]:
        return out

    # 3.【事实闸】结论极性不许和池子矛盾。放在打分前面，省钱
    out["facts_ok"] = _facts_ok(natural, record, pool_answers)
    if not out["facts_ok"]:
        return out

    # 4.【σ-可分门】粗筛 → 双评 → 查表判误差带是否错开
    score_clean, score_base = cleaner_scores(
        references, natural, base_think,
        k_screen=int(gates["k_screen"]), k_select=int(gates["k_select"]))
    out["score_clean"], out["score_base"] = score_clean, score_base
    if score_clean is None:
        return out
    out["selected"] = confident_cleaner(score_clean, score_base, float(gates["n_sigma"]))
    return out


def run(cfg: Config, *, limit: int = 0) -> dict[str, Path]:
    """跑第 4 步。

    :param cfg: 配置
    :param limit: 只处理前 N 题，0 = 全量
    :return: ``{train, eval}`` 两个路径
    """
    global _pool_index

    rows = read_jsonl(paths.train_set(cfg))
    if limit:
        rows = rows[:limit]
    _pool_index = load_pool_index(cfg)

    sft_cfg = cfg.train["sft"]
    target = int(sft_cfg["target_samples"])
    log.info("冷启动改写 + 选样：%d 题；答案池覆盖 %d 题；目标 %d 条",
             len(rows), len(_pool_index), target)

    results = gather_until(
        rows, lambda r: _process(cfg, r),
        enough=lambda rs: target > 0 and sum(1 for r in rs if r and r.get("selected")) >= target,
        chunk=int(cfg.data["gather_chunk"]),
        workers=int(cfg.judge["call_workers"]),
        desc="sft_data",
        progress_path=paths.progress(cfg, "sft"),
        seed=int(cfg.data["gather_seed"]))

    # 5.【留出验证集】按 qid 稳定排序后每隔 N 条抽一条。
    #    确定性留出，不用随机：续跑和跑满会得到同一批验证题，两次训练的 best checkpoint 才可比。
    eligible = [r for r in results if r.get("selected")]
    eligible.sort(key=lambda r: r.get("qid") or "")
    eval_fraction = float(sft_cfg["eval_fraction"])
    every = max(2, round(1 / eval_fraction)) if eval_fraction > 0 else 0

    train_rows, eval_rows = [], []
    for index, item in enumerate(eligible):
        row = sft_row(item["user_prompt"], item["natural"], item["answer"], query=item.get("query"))
        if every and index % every == 0:
            eval_rows.append(row)
        else:
            train_rows.append(row)

    write_jsonl(paths.sft_train(cfg), train_rows)
    write_jsonl(paths.sft_eval(cfg), eval_rows)

    # 6.【漏斗日志】每道门各挡掉多少。产出率一旦异常，看这一行就知道卡在哪道门上
    n_rewritten = sum(1 for r in results if r.get("natural"))
    n_rule_blocked = sum(1 for r in results if r.get("natural") and not r.get("rule_ok"))
    n_facts_blocked = sum(1 for r in results if r.get("rule_ok") and not r.get("facts_ok"))
    n_screened = sum(1 for r in results if r.get("facts_ok") and r.get("score_clean") is None)
    log.info("漏斗：改写成功 %d / 规则门挡 %d / 事实闸挡 %d / 粗筛挡 %d / 可用 %d（留出验证 %d）",
             n_rewritten, n_rule_blocked, n_facts_blocked, n_screened, len(eligible), len(eval_rows))
    log.info("train=%d -> %s ; eval=%d -> %s",
             len(train_rows), paths.sft_train(cfg), len(eval_rows), paths.sft_eval(cfg))

    if 0 < len(eligible) < target:
        log.warning("找遍/早停后只凑到 %d 条 < 目标 %d，用现有这些继续训。", len(eligible), target)
    if not train_rows:
        log.warning("一条合格样本都没有。查改写质量、规则门命中、σ 阈值，或把 gates.n_sigma 调小。")
    elif len(train_rows) < _MIN_USABLE_SAMPLES:
        log.warning("训练桶只有 %d 条 < %d：样本太少，SFT 学不到稳定写法。",
                    len(train_rows), _MIN_USABLE_SAMPLES)

    return {"train": paths.sft_train(cfg), "eval": paths.sft_eval(cfg)}
