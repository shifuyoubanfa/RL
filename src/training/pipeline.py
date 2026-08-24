"""训练编排：把四个阶段串成一条线，每一步都能断点续跑。

本模块在整条链路里的位置：总控。`scripts/train.sh` 最终执行的就是它。

一条主线，五个训练产物：

    基座（冻结）
      └─ sft          冷启动，学会自然推理这种写法
         └─ rft       拒绝采样，模型自采自筛自训，把 grounding 拉回来
            └─ dpo    偏好对，第一次真正的偏好学习
               └─ grpo-warmup   在线预热，只用规则硬门稳格式
                  └─ grpo-online 在线强化，接裁判在安全区内排序

每个阶段固定四拍：**建数据 → 训 LoRA → 合并成全量 → 三件套评测**。

**GPU 是互斥的。** 采样和评测需要一个常驻推理服务占着卡，训练和合并需要卡是空的。
所以编排器在每次训练前必须显式把推理服务停掉，训练结束再按需起回来。
这件事不做，第二个阶段起就会 OOM，而且报错位置在框架深处，看不出是资源冲突。

**续跑靠标记文件，不靠"输出在不在"。** 选样一条都没选中会写出 0 字节文件，
进程被 kill 也会留下不完整的文件，两者用文件大小分不开。所以每一步整步成功才写 `.done`，
续跑只认它。
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import Config, ensure_dirs, load_config
from src.data import (build_answer_pool, build_base_outputs, build_dpo_pairs, build_grpo_data,
                      build_rft_data, build_sft_data, paths, rollout, split_dataset)
from src.data.jsonl_io import nonempty
from src.evaluation import infer as eval_infer
from src.evaluation import metrics as eval_metrics
from src.models import vllm_client
from src.utils.log import get_logger

log = get_logger("pipeline")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"

# 阶段名 → 它在推理服务里对外的名字。这个名字决定套哪套 system 提示，见 prompts.system_for。
_STAGE_ORDER = ("sft", "rft", "dpo", "grpo-warmup", "grpo-online")

# 停推理服务后，最多等这么多轮（每轮 5 秒）确认端口真的空了。
_STOP_POLL_ROUNDS = 12
_STOP_POLL_INTERVAL = 5
# 起服务后等就绪的上限。32B 权重加载 + 并行初始化，慢盘上要好几分钟。
_SERVE_READY_TIMEOUT = 1800


# ---------------------------------------------------------------------------
# 状态与子进程
# ---------------------------------------------------------------------------

class Runner:
    """跑子进程、记事件、存状态。

    每一步都是独立子进程（训练脚本是 bash，数据步是 python），日志各自落一个文件。
    这里额外维护一份 `state.json`，让人在另一个终端 `cat` 一下就知道现在跑到哪了、
    该看哪个日志——长跑任务里这比翻几万行日志有用得多。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.raw_dir = Path(cfg.paths["log_dir"]) / "pipeline"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.raw_dir / "state.json"
        self.event_log = self.raw_dir / "events.log"
        self.state: dict = {"status": "starting", "stage": "preflight", "completed": []}

    def emit(self, message: str) -> None:
        """打一条事件，同时进控制台和事件日志。"""
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {message}"
        print(line, flush=True)
        with self.event_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_state(self, **updates) -> None:
        """原子更新状态文件。写 .tmp 再 replace，避免别人读到写了一半的 JSON。"""
        self.state.update(updates, updated_at=f"{datetime.now():%Y-%m-%d %H:%M:%S}")
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_file)

    def run(self, stage: str, cmd: list, *, env: dict | None = None) -> None:
        """跑一个子进程，全程把输出写进该阶段的日志文件。

        :param stage: 阶段名，决定日志文件名
        :param cmd: 命令行
        :param env: 额外的环境变量
        :raise SystemExit: 子进程返回非 0
        """
        raw_log = self.raw_dir / f"{stage}.log"
        cmd = [str(c) for c in cmd]
        full_env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
        self.emit(f"START {stage}")
        self.emit("CMD   " + " ".join(cmd))

        with raw_log.open("a", encoding="utf-8") as sink:
            sink.write(f"\n===== START {' '.join(cmd)} =====\n")
            sink.flush()
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=full_env,
                                    stdout=sink, stderr=subprocess.STDOUT, text=True)
            self.save_state(stage=stage, status="running", pid=proc.pid, raw_log=str(raw_log))
            started = time.time()
            # 每分钟打一次心跳。训练几个小时不出声，人会以为挂了去 kill 它。
            while proc.poll() is None:
                time.sleep(30)
                if int(time.time() - started) % 60 < 30:
                    self.emit(f"{stage} | 运行中 pid={proc.pid} 已 {int(time.time() - started)}s")
            code = proc.returncode
            sink.write(f"===== END rc={code} =====\n")

        if code != 0:
            self.save_state(status="failed", returncode=code)
            self.emit(f"FAIL  {stage} rc={code}；日志见 {raw_log}")
            raise SystemExit(code)
        if stage not in self.state["completed"]:
            self.state["completed"].append(stage)
        self.save_state(status="running")
        self.emit(f"END   {stage}")


# ---------------------------------------------------------------------------
# 完成判定 / 断点续跑
# ---------------------------------------------------------------------------

def _marked(cfg: Config, name: str) -> bool:
    """这一步整步做完了没有。"""
    return paths.stage_marker(cfg, name).exists()


def _mark(cfg: Config, name: str) -> None:
    """写整步完成标记。"""
    marker = paths.stage_marker(cfg, name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}", encoding="utf-8")


def _resolve_adapter(root: Path) -> Path:
    """在 LoRA 输出目录下找到真正含 `adapter_config.json` 的那一层。

    训练框架会把 adapter 存进 `<root>/vN-时间戳/checkpoint-N/`，层数和名字随版本变。
    优先读 `trainer_state.json` 里记的 best checkpoint；没有就取编号最大的那个。
    两条都不成立就原样返回 root，让下游报一个清楚的"找不到 adapter_config.json"。

    :param root: LoRA 输出目录
    :return: adapter 目录
    """
    import re

    def checkpoint_number(path) -> int:
        match = re.search(r"checkpoint-(\d+)", str(path))
        return int(match.group(1)) if match else -1

    states = list(root.glob("**/trainer_state.json"))
    if states:
        try:
            newest = max(states, key=checkpoint_number)
            best = json.loads(newest.read_text(encoding="utf-8")).get("best_model_checkpoint")
            if best and (Path(best) / "adapter_config.json").exists():
                return Path(best)
        except Exception:                              # noqa: BLE001 - 读不出 best 就退回"取编号最大"
            pass
    configs = list(root.glob("**/adapter_config.json"))
    return max(configs, key=checkpoint_number).parent if configs else root


def _adapter_done(root: Path) -> bool:
    """LoRA 训完了没有。

    以 `.done` 标记为准，**不用 glob adapter_config.json**：训练每个 epoch 都会落一个
    checkpoint，中途崩溃会留下半成品，glob 会把半成品当成"已训完"直接跳过训练，
    拿一份没收敛的权重进下游。

    :raise RuntimeError: 有 .done 却找不到 adapter，说明目录被人动过
    """
    if not (root / ".done").exists():
        return False
    if not (_resolve_adapter(root) / "adapter_config.json").exists():
        raise RuntimeError(f"有完成标记但找不到 adapter: {root}")
    return True


def _merged_done(path: Path) -> bool:
    """合并出来的全量模型完整不完整。

    :raise RuntimeError: 有 .done 却缺 config 或权重分片
    """
    if not (path / ".done").exists():
        return False
    if not (path / "config.json").exists() or not any(path.glob("*.safetensors")):
        raise RuntimeError(f"有完成标记但模型不完整: {path}")
    return True


def _move_interrupted(runner: Runner, path: Path) -> None:
    """把没有完成标记的残留输出挪走，不删除。

    这些目录动辄几十 G，重跑要几十分钟。自动删一次删错，代价太大。
    """
    if path.exists() and not (path / ".done").exists():
        moved = path.with_name(f"{path.name}.interrupted-{datetime.now():%Y%m%d-%H%M%S}")
        runner.emit(f"保留中断的输出: {path} -> {moved}")
        path.rename(moved)


# ---------------------------------------------------------------------------
# 推理服务起停
# ---------------------------------------------------------------------------

def stop_vllm(runner: Runner) -> None:
    """停掉推理服务，并确认端口真的空了。

    按进程组 kill：张量并行会拉起一堆 worker 子进程，只 kill 主进程的话
    worker 还占着显存，下一步训练照样 OOM。

    :raise RuntimeError: 已知进程组都停了，端口还在服务——说明有别人起的服务占着卡
    """
    pid_file = Path(runner.cfg.paths["log_dir"]) / "vllm.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            runner.emit(f"停止推理服务进程组 {pid}")
            os.killpg(pid, signal.SIGTERM)
            time.sleep(8)                              # 给它一点时间优雅退出并释放显存
            # 已经自己退干净了就没什么好 kill 的
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
        except (ValueError, ProcessLookupError, PermissionError, AttributeError):
            # AttributeError：Windows 上没有 killpg。这条链路的训练只在 Linux 上跑，
            # 本地跑数据步时没有服务要停，忽略即可。
            pass
        pid_file.unlink(missing_ok=True)

    for _ in range(_STOP_POLL_ROUNDS):
        if not vllm_client.health():
            return
        time.sleep(_STOP_POLL_INTERVAL)
    raise RuntimeError("推理端口仍在服务：有本编排器之外的进程占着卡，先手工处理")


def serve_model(runner: Runner, model_dir: Path, served_name: str) -> None:
    """起推理服务并等就绪。起之前先把旧的停掉。"""
    stop_vllm(runner)
    runner.run(f"serve_{served_name}",
               ["bash", str(SCRIPTS / "serve_vllm.sh"), str(model_dir), served_name])
    runner.emit(f"等待推理服务就绪 {served_name} ({model_dir})")
    vllm_client.wait_ready(max_wait=_SERVE_READY_TIMEOUT)
    runner.emit(f"推理服务就绪 {served_name}")


# ---------------------------------------------------------------------------
# 训练环境变量
# ---------------------------------------------------------------------------

def _train_env(cfg: Config, extra: dict | None = None) -> dict:
    """把配置里的训练超参翻译成训练脚本认的环境变量。

    集中在一处翻译，是为了让"改 yaml 就能改训练行为"这句话成立。
    散在各处拼命令行，迟早会有一个参数只在某个脚本里写死。

    :param cfg: 配置
    :param extra: 该阶段特有的变量
    :return: 环境变量字典
    """
    train = cfg.train
    env = {
        "TRAIN_GPUS": str(train["gpus"]),
        "DEEPSPEED": str(train["deepspeed"]),
        "MODEL_TYPE": str(cfg.model["model_type"]),
        "TEMPLATE": str(cfg.model["template"]),
        "LORA_R": str(train["lora_rank"]),
        "LORA_ALPHA": str(train["lora_alpha"]),
        "LORA_DROPOUT": str(train["lora_dropout"]),
        "TARGET_MODULES": str(train["target_modules"]),
        "MAX_LEN": str(train["max_length"]),
    }
    env.update(extra or {})
    return env


# ---------------------------------------------------------------------------
# 四拍：建数据 / 训练 / 合并 / 评测
# ---------------------------------------------------------------------------

def train_lora(runner: Runner, script: str, stage: str, args: list, env: dict) -> Path:
    """训一个 LoRA。已经训完就跳过。

    训练前一定先停推理服务——训练要独占卡。

    :return: LoRA 输出目录
    """
    out_dir = paths.lora_dir(runner.cfg, stage)
    if _adapter_done(out_dir):
        runner.emit(f"SKIP  {stage} 训练；已完成: {out_dir}")
        return out_dir
    _move_interrupted(runner, out_dir)
    stop_vllm(runner)
    runner.run(f"train_{stage}", ["bash", str(SCRIPTS / "swift" / script), *args, str(out_dir)], env=env)
    return out_dir


def merge_lora(runner: Runner, stage: str, base: Path) -> Path:
    """把某阶段的 LoRA 合并回基座。已经合并过就跳过。

    :return: 合并出来的全量模型目录
    """
    out_dir = paths.merged_dir(runner.cfg, stage)
    if _merged_done(out_dir):
        runner.emit(f"SKIP  {stage} 合并；已存在: {out_dir}")
        return out_dir
    _move_interrupted(runner, out_dir)
    adapter = _resolve_adapter(paths.lora_dir(runner.cfg, stage))
    stop_vllm(runner)
    runner.run(f"merge_{stage}",
               ["bash", str(SCRIPTS / "merge_lora.sh"), str(base), str(adapter), str(out_dir)],
               env=_train_env(runner.cfg))
    return out_dir


def evaluate(runner: Runner, tag: str, model_dir: Path, served_name: str) -> dict:
    """跑一次三件套评测。已经评过就直接读摘要。

    推理阶段占卡，判分阶段只要网络，所以推理一结束就把服务停掉。

    :param tag: 评测标签，决定产物文件名
    :param model_dir: 要评的全量模型
    :param served_name: 在推理服务里对外的名字，决定套哪套 system 提示
    :return: 机读摘要
    """
    cfg = runner.cfg
    infer_path, scores_path, report_path, summary_path = paths.eval_paths(cfg, tag)
    if eval_metrics.report_is_complete(report_path, summary_path):
        runner.emit(f"SKIP  评测 {tag}；报告已完整: {report_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    serve_model(runner, model_dir, served_name)
    eval_infer.run(cfg, model_name=served_name, eval_file=paths.eval_set(cfg), out_path=infer_path)
    stop_vllm(runner)
    return eval_metrics.run(cfg, infer_path=infer_path, scores_path=scores_path,
                            report_path=report_path, summary_path=summary_path, tag=tag)


def check_stage_gate(runner: Runner, tag: str, summary: dict) -> None:
    """阶段门：答案在池率跌破地板就停。

    只用绝对地板，不和上一阶段做差。原因是相邻两个阶段的 system 提示可能不同
    （基座用 RAG 腔，训练后的模型用中性提示），跨提示词的差值不可比——
    健康的去检索腔模型本来就会比基座的在池率低几个点，拿差值当判据会误停。

    地板是人定的容差，不是统计判据：在池率由确定性规则算出，没有打分噪声。

    :raise SystemExit: 跌破地板
    """
    floor = float(runner.cfg.gates["in_pool_floor"])
    rate = float(summary.get("in_pool_rate", 0.0))
    if rate < floor:
        runner.emit(f"阶段门失败 {tag}: 答案在池率 {rate:.1%} < 地板 {floor:.0%}")
        raise SystemExit(
            f"{tag} 的答案在池率跌破地板，说明答案开始漂了。"
            f"先查这一阶段的数据（选样门是不是放太松），别接着往下训。")
    runner.emit(f"阶段门通过 {tag}: 答案在池率 {rate:.1%} >= 地板 {floor:.0%}")


# ---------------------------------------------------------------------------
# 五个阶段
# ---------------------------------------------------------------------------

def build_upstream(runner: Runner) -> None:
    """上游三步：基座重产 → 切分 → 建答案池。只跑一次，后面所有阶段共用。"""
    cfg = runner.cfg
    base_model = Path(cfg.model["base_model"])

    if not _marked(cfg, "base_outputs"):
        serve_model(runner, base_model, str(cfg.vllm["served_model"]))
        build_base_outputs.run(cfg)
        _mark(cfg, "base_outputs")
    if not _marked(cfg, "split"):
        split_dataset.run(cfg)
        _mark(cfg, "split")
    if not _marked(cfg, "answer_pool"):
        serve_model(runner, base_model, str(cfg.vllm["served_model"]))
        build_answer_pool.run(cfg)
        _mark(cfg, "answer_pool")
    stop_vllm(runner)


def stage_sft(runner: Runner, base_model: Path) -> Path:
    """阶段一：冷启动 SFT。"""
    cfg = runner.cfg
    if not _marked(cfg, "sft_data"):
        stop_vllm(runner)                              # 改写和打分只走裁判服务，不占卡
        build_sft_data.run(cfg)
        _mark(cfg, "sft_data")
    if not nonempty(paths.sft_train(cfg)):
        raise SystemExit("冷启动训练桶是空的，选样门一条都没选中。查漏斗日志，或把 gates.n_sigma 调小。")
    if not nonempty(paths.sft_eval(cfg)):
        raise SystemExit("冷启动验证集是空的，训练框架的 load_best 会崩。先查 build_sft_data 的留出逻辑。")

    sft = cfg.train["sft"]
    train_lora(runner, "sft.sh", "sft",
               [str(paths.sft_train(cfg)), str(paths.sft_eval(cfg)), str(base_model),
                str(sft["learning_rate"]), str(sft["epochs"])],
               _train_env(cfg, {"PDBS": str(sft["per_device_batch_size"]),
                                "GA": str(sft["gradient_accumulation"]),
                                "SAVE_TOTAL_LIMIT": str(sft["save_total_limit"])}))
    return merge_lora(runner, "sft", base_model)


def stage_rft(runner: Runner, sft_merged: Path) -> Path:
    """阶段二：拒绝采样。先用刚训好的模型自采，再筛，再训。"""
    cfg = runner.cfg
    rft = cfg.train["rft"]
    if not _marked(cfg, "rft_data"):
        if not nonempty(paths.rft_samples(cfg)):
            serve_model(runner, sft_merged, "sft")
            rollout.sample_candidates(cfg, problems_path=paths.problems_train(cfg),
                                      out_path=paths.rft_samples(cfg),
                                      model_name="sft", k=int(rft["selfsample_k"]))
        stop_vllm(runner)
        build_rft_data.run(cfg)
        _mark(cfg, "rft_data")
    if not nonempty(paths.rft_train(cfg)):
        raise SystemExit("拒绝采样训练桶是空的。模型自采可能已普遍偏干净、裁判分不开；"
                         "可以跳过这一阶段直接做偏好对，或把 gates.n_sigma 调小。")

    train_lora(runner, "sft.sh", "rft",
               [str(paths.rft_train(cfg)), str(paths.sft_eval(cfg)), str(sft_merged),
                str(rft["learning_rate"]), str(rft["epochs"])],
               _train_env(cfg, {"PDBS": str(cfg.train["sft"]["per_device_batch_size"]),
                                "GA": str(cfg.train["sft"]["gradient_accumulation"])}))
    return merge_lora(runner, "rft", sft_merged)


def stage_dpo(runner: Runner, rft_merged: Path) -> Path:
    """阶段三：偏好优化。"""
    cfg = runner.cfg
    dpo = cfg.train["dpo"]
    if not _marked(cfg, "dpo_data"):
        if not nonempty(paths.dpo_rollout(cfg)):
            serve_model(runner, rft_merged, "rft")
            rollout.sample_candidates(cfg, problems_path=paths.train_set(cfg),
                                      out_path=paths.dpo_rollout(cfg),
                                      model_name="rft", k=int(dpo["rollout_k"]))
        stop_vllm(runner)
        build_dpo_pairs.run(cfg)
        _mark(cfg, "dpo_data")
    if not nonempty(paths.dpo_pairs(cfg)):
        raise SystemExit("偏好对是空的。查 rollout 质量，或把 gates.n_sigma 调小。")

    train_lora(runner, "dpo.sh", "dpo",
               [str(paths.dpo_pairs(cfg)), str(rft_merged)],
               _train_env(cfg, {"DPO_BETA": str(dpo["beta"]), "DPO_LR": str(dpo["learning_rate"]),
                                "DPO_EPOCHS": str(dpo["epochs"]), "DPO_GA": str(dpo["gradient_accumulation"]),
                                "DPO_RPO_ALPHA": str(dpo["rpo_alpha"]),
                                "DPO_SAVE_STEPS": str(dpo["save_steps"]),
                                "DPO_SAVE_TOTAL_LIMIT": str(dpo["save_total_limit"])}))
    return merge_lora(runner, "dpo", rft_merged)


def _grpo_env(cfg: Config, phase: dict) -> dict:
    """拼 GRPO 一段的环境变量。warmup 和 online 只有几个值不同，共用这一个函数。"""
    grpo = cfg.train["grpo"]
    return _train_env(cfg, {
        "GRPO_K": str(grpo["group_size"]),
        "GRPO_STEPS": str(phase["steps"]),
        "GRPO_LR": str(phase["learning_rate"]),
        "GRPO_BETA": str(phase["beta"]),
        "GRPO_REWARD_FUNC": str(phase["reward_func"]),
        "GRPO_TEMPERATURE": str(grpo["temperature"]),
        "GRPO_TOP_P": str(grpo["top_p"]),
        "GRPO_MAX_COMPLETION": str(grpo["max_completion_length"]),
        "GRPO_SAVE_STEPS": str(grpo["save_steps"]),
        "GRPO_SAVE_TOTAL_LIMIT": str(grpo["save_total_limit"]),
        "SCALE_REWARDS": str(grpo["scale_rewards"]),
        "VLLM_GPU_UTIL": str(grpo["vllm_gpu_util"]),
        "VLLM_MAX_LEN": str(grpo["vllm_max_len"]),
        "MOVE_MODEL_BATCHES": str(grpo["move_model_batches"]),
        "SLEEP_LEVEL": str(grpo["sleep_level"]),
        "OFFLOAD_MODEL": str(grpo["offload_model"]).lower(),
        "OFFLOAD_OPTIMIZER": str(grpo["offload_optimizer"]).lower(),
        "PDBS": str(grpo["per_device_batch_size"]),
        "GA": str(grpo["gradient_accumulation"]),
    })


def stage_grpo(runner: Runner, dpo_merged: Path) -> Path:
    """阶段四：在线 GRPO，两段课程式。

    先跑 warmup（只用规则硬门），合并，再跑 online（接裁判）。
    中间那次合并不能省：第二段的参考模型必须是第一段的结果，
    否则 KL 约束是拿"没预热过的模型"当锚，等于把预热学到的东西又往回拉。
    """
    cfg = runner.cfg
    grpo = cfg.train["grpo"]
    if not _marked(cfg, "grpo_data"):
        stop_vllm(runner)
        build_grpo_data.run(cfg)
        _mark(cfg, "grpo_data")

    current = dpo_merged
    if int(grpo["warmup"]["steps"]) > 0:
        train_lora(runner, "grpo.sh", "grpo-warmup",
                   [str(paths.grpo_data(cfg)), str(current)], _grpo_env(cfg, grpo["warmup"]))
        current = merge_lora(runner, "grpo-warmup", current)

    train_lora(runner, "grpo.sh", "grpo-online",
               [str(paths.grpo_data(cfg)), str(current)], _grpo_env(cfg, grpo["online"]))
    return merge_lora(runner, "grpo-online", current)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def preflight(runner: Runner) -> None:
    """开跑前的检查。全在占用 GPU 之前做完。

    :raise SystemExit: 任一前置条件不满足
    """
    cfg = runner.cfg
    from src.config import judge_api_key

    missing = [str(p) for p in (
        Path(cfg.model["base_model"]) / "config.json",
        SCRIPTS / "swift" / "sft.sh",
        SCRIPTS / "swift" / "dpo.sh",
        SCRIPTS / "swift" / "grpo.sh",
        SCRIPTS / "merge_lora.sh",
        SCRIPTS / "serve_vllm.sh",
    ) if not Path(p).exists()]
    if missing:
        raise SystemExit("preflight 缺文件：\n" + "\n".join(missing))

    judge_api_key(cfg)                                 # key 没设就在这里报，不要训到一半才发现
    stop_vllm(runner)
    runner.emit(f"preflight OK | base={cfg.model['base_model']} | "
                f"n_sigma={cfg.gates['n_sigma']} | 在池率地板={cfg.gates['in_pool_floor']}")


def run_pipeline(cfg: Config, *, stages: tuple[str, ...] = _STAGE_ORDER,
                 eval_baseline: bool = True) -> None:
    """跑主线。

    :param cfg: 配置
    :param stages: 要跑哪几个阶段。默认全跑；调试时可以只跑前面几个
    :param eval_baseline: 要不要先评一次基座基线
    """
    ensure_dirs(cfg)
    runner = Runner(cfg)
    runner.save_state(status="running", stage="preflight", completed=[])
    base_model = Path(cfg.model["base_model"])

    try:
        preflight(runner)
        build_upstream(runner)

        if eval_baseline:
            # 基座基线必须用它自己的名字 serve，才会套上 RAG 腔提示、还原它真实的分布。
            evaluate(runner, "baseline", base_model, str(cfg.vllm["served_model"]))

        current = base_model
        if "sft" in stages:
            current = stage_sft(runner, current)
            check_stage_gate(runner, "sft", evaluate(runner, "sft", current, "sft"))
        if "rft" in stages:
            current = stage_rft(runner, current)
            check_stage_gate(runner, "rft", evaluate(runner, "rft", current, "rft"))
        if "dpo" in stages:
            current = stage_dpo(runner, current)
            check_stage_gate(runner, "dpo", evaluate(runner, "dpo", current, "dpo"))
        if "grpo-online" in stages:
            current = stage_grpo(runner, current)
            check_stage_gate(runner, "grpo", evaluate(runner, "grpo", current, "grpo"))

        runner.save_state(status="complete", stage="done", pid=None, final_model=str(current))
        runner.emit(f"主线跑完 | 最终模型 = {current}")
    except BaseException as exc:
        runner.save_state(status="failed", error=repr(exc))
        runner.emit(f"主线失败 | {exc!r}")
        raise
    finally:
        try:
            stop_vllm(runner)
        except Exception as exc:                       # noqa: BLE001 - 收尾清理失败只告警，不掩盖真正的失败原因
            runner.emit(f"WARN 收尾停推理服务失败: {exc!r}")


def main() -> None:
    """命令行入口。由 `scripts/train.sh` 调用。"""
    import argparse

    parser = argparse.ArgumentParser(description="跑强化学习训练主线")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 configs/train.yaml")
    parser.add_argument("--stages", default="all",
                        help=f"要跑哪些阶段，逗号分隔。可选：{','.join(_STAGE_ORDER)}；all = 全跑")
    parser.add_argument("--no-baseline", action="store_true", help="跳过基座基线评测")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stages = _STAGE_ORDER if args.stages == "all" else tuple(
        s.strip() for s in args.stages.split(",") if s.strip())
    unknown = [s for s in stages if s not in _STAGE_ORDER]
    if unknown:
        raise SystemExit(f"未知阶段 {unknown}；可选：{list(_STAGE_ORDER)}")
    run_pipeline(cfg, stages=stages, eval_baseline=not args.no_baseline)


if __name__ == "__main__":
    sys.exit(main())
