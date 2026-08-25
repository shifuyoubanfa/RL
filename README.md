# 32B 模型推理过程去检索腔 · 强化学习训练链路

用四阶段强化学习，把一个 32B 领域模型的推理过程从"念手册、查资料"改造成"像人一步步推"，
同时锁死最终答案不漂离该模型自己原本给出的结论。

产出是一条可复跑的端到端链路：**数据构建 → 训练 → 评测**，三个入口，一份配置。

---

## 一、这条链路解决什么问题

有一个 32B 规模的领域模型。它对外宣称做端到端推理，但它的 `<think>` 段里满是检索腔——
"根据参考问答对1"、"资料显示"、"检索结果表明"。读起来不像在推理，像在念检索到的材料。

要改的就是这个。约束有两条：

1. **不能蒸馏。** 这个领域没有比它更强的老师模型可用。所以只有"让它自己采样、自己筛好样本、
   再训自己"这一条路。
2. **答案一个字不许动。** 只改推理过程的写法，最终答复必须还在它自己认可的范围内。
   推理洗干净了但答案变了，等于把产品做坏。

第二条约束贯穿全链路，代码里叫 **answer-lock**：所有训练样本的 `<answer>` 段都拼基座原版，
只有 `<think>` 段是新的。梯度因此只压在推理段上。

---

## 二、核心训练流程

```
原始语料
  → 基座重产 think/answer          （拿到答案金标准 + 待改造的推理）
  → 切训练集 / 冻结验收集
  → 建基座认可答案池                （每题贪心1条 + 采样N条，当漂移判定靶子）
  → 冷启动 SFT     教会"自然推理"这种写法
  → 拒绝采样 SFT   模型自采、自筛、自训，把 grounding 拉回来
  → DPO 偏好对     第一次真正的偏好学习
  → 在线 GRPO      两段课程式：规则预热 → 接裁判在线排序
  → 三件套离线评测
```

每个阶段固定四拍：**建数据 → 训 LoRA → 合并成全量模型 → 三件套评测 → 过阶段门**。

四个阶段实际跑出来的量级（完整追溯见 [docs/RESULTS.md](docs/RESULTS.md) 第 4 节）：

| 阶段 | 数据怎么来 | 实得 | 训练配置 |
|---|---|---:|---|
| 冷启动 SFT | 裁判改写 1739 题 → 三道门 | 930 条 | LoRA r16/α32，lr 5e-5，7 轮 ≈ 100 步 |
| 拒绝采样 | 每题自采 32 条 → 三道门 | 204 条 | lr 3e-5，3 轮 |
| DPO | 每题采 16 条 → 构 answer-lock 对子 | 885 对 | β 0.1，lr 5e-6，2 轮，rpo_alpha 1.0，108 步 |
| 在线 GRPO | 1668 道可训练题，组大小 K=8 | — | 预热 30 步（β 0.08）+ 在线 90 步（β 0.06），lr 7e-7 |

用到的框架和各自的职责：

| 框架 | 在这里干什么 |
|---|---|
| **ms-swift** | 训练侧全部。`swift sft` 跑监督微调，`swift rlhf --rlhf_type dpo/grpo` 跑偏好优化和在线强化，`swift export` 合并 LoRA |
| **LoRA (PEFT)** | 32B 全参训不动，四个阶段全走 LoRA，只训注意力和 MLP 的七个投影矩阵 |
| **DeepSpeed ZeRO-3** | 把参数、梯度、优化器状态切到各卡上，让 32B 能在多卡上训起来 |
| **vLLM** | 采样和评测的推理服务。数据构建、rollout、评测推理都走它的 HTTP 接口；在线 GRPO 里它以 colocate 模式住在训练进程内 |
| **一个外部裁判模型** | 唯一由大模型判的事：这段推理是不是在"换词复述参考资料"。规则看不见这个 |

---

## 三、最终成绩

全部数字在**同一套冻结的 500 题验收集**上量，每个都标了"谁评的 + 样本量"。
完整口径、数据流追溯、裁判标定全表见 [docs/RESULTS.md](docs/RESULTS.md)。

| 阶段 | 裁判 think 干净分<br>0–10，**裁判 k=3**，N=500 | 规则去检索腔通过率<br>**规则**，N=500 | 答案在池率<br>**规则**，N=500 |
|---|---:|---:|---:|
| 基座基线 | 3.140 | 2.6% | 93.8% |
| 冷启动 SFT | 4.408 | 45.4% | 83.0% |
| 拒绝采样 RFT | 4.489 | 45.6% | 85.0% |
| **DPO ★最优** | **4.643** | 49.0% | 84.6% |
| 在线 GRPO | 4.601 | **50.0%** | 83.8% |

**真涨门。** 干净分是裁判打的、有噪声：N=500、k=3 时 SE ≈ 0.0495，
所以两阶段之差要超过 **3×SE ≈ 0.15** 才算真涨，否则记统计打平。
另外两列是确定性规则算的，没有打分噪声。

四步里只有两步越过了这道门：

- **基座 → SFT：干净分 +1.268，同时规则通过率 2.6% → 45.4%。** 主观分和客观率同向大涨，
  排除了"只把代理指标刷高"的可能。
- **RFT → DPO：干净分 +0.155，刚越过门。** 这是这条 RL 线唯一一次量出偏好学习真的把
  推理写法学干净了。

**DPO 与 GRPO 统计打平**——三件套差值全部落在各自噪声门槛内，最优仍记 DPO。
在线阶段的定位是守住增益并确认天花板，不是再上一个台阶，原因见 RESULTS 第 3 节。

**⚠️ 三件套整体判 FAIL，唯一卡点是答案在池率 84.6%，差 0.4 个点没回到 85% 地板。**
这是一个明确的、可以继续工程化的剩余问题：answer-lock 锁的是训练样本里的答案文本，
而推理时模型是现吐一段新答案的，把推理段推干净的策略顺带把答案分布也推偏了一点。

---

## 四、目录结构

```
.
├── configs/train.yaml          全流程唯一配置：路径 / 超参 / 门限 / 标定表
├── scripts/
│   ├── prepare_data.py         数据处理入口（七个子步骤）
│   ├── train.sh                训练入口（--stages 选阶段）
│   ├── evaluate.py             评测入口（跑分 / 比较两个阶段）
│   ├── serve_vllm.sh           起推理服务
│   ├── merge_lora.sh           合并 LoRA 回基座
│   └── swift/{sft,dpo,grpo}.sh 三个训练启动脚本，参数带中文注释
├── src/
│   ├── config.py               读配置 + 环境变量覆盖
│   ├── data/                   数据构建：重产、切分、建池、四阶段各自的数据
│   ├── models/                 推理服务客户端、裁判服务客户端
│   ├── rewards/                规则层、裁判打分、标定表、在线 GRPO 奖励函数
│   ├── training/pipeline.py    主线编排：串四个阶段、断点续跑、阶段门
│   ├── evaluation/             评测推理 + 三件套判分
│   └── utils/                  日志、并发执行器
├── examples/
│   ├── sample_data.jsonl       输入语料格式（合成数据）
│   └── sample_train_rows.jsonl 三种训练样本格式（由代码生成）
├── tests/                      离线契约测试，纯 CPU，不连任何服务
└── docs/
    ├── RESULTS.md              全部数字的唯一出处：成绩、数据流追溯、裁判标定表
    └── interview_guide.md      逐段讲解：每一步做什么、为什么这么做
```

---

## 五、环境安装

编排层和数据层的依赖很轻，装完就能跑数据处理、评测和全部测试，**不需要 GPU**：

```bash
pip install -r requirements.txt
```

训练和推理的重依赖不在 `requirements.txt` 里，因为它们通常装在不同的 conda 环境：

```bash
# 训练环境：ms-swift 会带上 torch / transformers / peft / trl / deepspeed
conda create -n rl-train python=3.10 && conda activate rl-train
pip install ms-swift deepspeed

# 推理环境：vLLM 和训练侧的 torch 版本容易打架，单独建一个
conda create -n rl-serve python=3.10 && conda activate rl-serve
pip install vllm
```

两个环境的可执行文件用环境变量指过去（见 `.env.example`）：

```bash
export SWIFT_BIN=/opt/conda/envs/rl-train/bin/swift
export VLLM_BIN=/opt/conda/envs/rl-serve/bin/vllm
export PYTHON_BIN=/opt/conda/envs/rl-train/bin/python
export DASHSCOPE_API_KEY=<裁判服务的 key>
```

---

## 六、数据格式

**输入语料**（`paths.raw_corpus`）每行两个字段，样例见 [examples/sample_data.jsonl](examples/sample_data.jsonl)：

```json
{"query": "小规模纳税人本月销售额9万元，需要缴纳增值税吗？",
 "user_prompt": "【参考问答对】\n问：…\n答：…\n\n【问题】\n小规模纳税人本月销售额9万元，需要缴纳增值税吗？"}
```

`user_prompt` 是完整题面，里面必须含 `【参考问答对】` 和 `【问题】` 两个标记——
参考资料段就是靠它们切出来的，裁判判"有没有在复述参考"要用。

**三种训练样本格式**见 [examples/sample_train_rows.jsonl](examples/sample_train_rows.jsonl)，
由 `src/data/schema.py` 直接生成，和代码永远一致：

| 格式 | 用在哪 | 特点 |
|---|---|---|
| `sft` | 冷启动、拒绝采样 | `messages` 三段，assistant 段是 `<think>` + `<answer>` |
| `dpo` | 偏好对 | chosen 在 `messages` 末尾，rejected 在顶层 `rejected_response`；**两边 answer 段逐字相同** |
| `grpo` | 在线强化 | 没有 assistant 段（答案训练时现场采），带 `v1_answers_json` 供奖励函数做硬门 |

**中间产物**全部落在 `paths.output_dir` 下，文件名前缀就是它在链路里的顺序：

```
00_base_outputs.jsonl     基座重产的 think/answer
01_train.jsonl            训练集
01_eval.jsonl             冻结验收集（全程不参与训练和选样）
02_answer_pool.jsonl      基座认可答案池
10_sft_train.jsonl        冷启动训练集      10_sft_eval.jsonl  留出验证集
20_rft_samples.jsonl      自采样原始结果     21_rft_train.jsonl 筛出来的训练集
30_dpo_rollout.jsonl      rollout 原始结果   31_dpo_pairs.jsonl 偏好对
40_grpo_prompts.jsonl     在线训练 prompt
eval/<tag>_{infer,scores,report,summary}.*   评测四件产物
```

---

## 七、训练命令

**跑完整主线**（建议放 tmux，长跑几十小时）：

```bash
tmux new -s rl
bash scripts/train.sh
```

它会按顺序做完：上游三步 → 基座基线评测 → 四个阶段各自的"建数据 → 训 → 合并 → 评测 → 过门"。

**只跑其中几个阶段**：

```bash
bash scripts/train.sh --stages sft,rft
bash scripts/train.sh --no-baseline          # 跳过基座基线
```

**单独跑某一步数据处理**（排查用）：

```bash
python scripts/prepare_data.py split
python scripts/prepare_data.py sft-data --limit 50
```

中断了直接重跑同一条命令即可。每一步整步成功才写完成标记，续跑只认标记，已完成的会跳过。

**进度**在另一个终端看：

```bash
cat runs/logs/pipeline/state.json     # 现在跑到哪一步、pid 多少、该看哪个日志
tail -f runs/logs/pipeline/events.log
```

---

## 八、评测命令

```bash
# 起服务后评一个模型
bash scripts/serve_vllm.sh /path/to/dpo-merged dpo
python scripts/evaluate.py --tag dpo --served-name dpo

# 推理结果已有，只重新判分
python scripts/evaluate.py --tag dpo --skip-infer

# 比较两个阶段，按真涨门判读
python scripts/evaluate.py --compare rft dpo
```

**三件套**，每个指标都标了谁评的：

| 指标 | 谁评的 | 判什么 | 有没有噪声 |
|---|---|---|---|
| 裁判干净分 | 裁判模型，每题打 k 遍 | 换词复述照抄的程度，0~10，越高越干净 | 有 |
| 规则通过率 | 确定性规则 | 推理段里有没有检索腔表面标记 | 无 |
| 答案在池率 | 确定性规则 | 答案的极性和数字还在不在基座认可池里 | 无 |

三个必须一起看。单看任何一个都能被刷高：只看干净分，模型可以写一段和参考毫无关系的空话；
只看规则通过率，不说那几个词就满分；只看在池率，原地不动就是满分。

干净分有噪声，两个阶段的差值要大过 **3 倍标准误**才算真涨，否则记统计打平。
门限由题数和 k 算出来，写在每份报告里。另外两个指标是确定性的，直接比数。

---

## 九、配置参数

改行为只改 [configs/train.yaml](configs/train.yaml)，不要改脚本。每一项都有中文注释。
换机器时至少要改这几个：

| 配置项 | 说明 |
|---|---|
| `paths.work_dir` | 产物根目录，下面挂 output / ckpts / models / logs |
| `paths.raw_corpus` | 原始语料 |
| `model.base_model` | 被强化的基座权重目录，必须含 `config.json` |
| `train.gpus` / `vllm.gpus` | 训练和推理各用哪几张卡 |
| `judge.api_key_env` | 裁判服务的 key 从哪个环境变量读 |

几个决定行为的旋钮：

| 配置项 | 作用 |
|---|---|
| `gates.n_sigma` | 选样门的严格程度。调大样本更干净，产出率掉得厉害 |
| `gates.k_screen / k_select / k_eval` | 三档判分遍数。`k_select` 必须 ≥ 16，否则标定表不作数 |
| `gates.in_pool_floor` | 阶段门的答案在池率地板，跌破就停 |
| `train.grpo.warmup.steps / online.steps` | 在线两段各跑多少步 |
| `judge.budget_yuan` | 裁判调用的花费围栏，0 = 只记账不设上限 |

环境变量可以覆盖常用项而不改文件，映射表在 `src/config.py` 顶部：

```bash
RL_WORK_DIR=/data/rl RL_SFT_TARGET=100 bash scripts/train.sh --stages sft
```

---

## 十、输出文件

| 位置 | 内容 |
|---|---|
| `output/` | 全部中间数据 jsonl（见第五节的文件名表） |
| `output/eval/<tag>_report.md` | 人读的三件套报告，每个数字带"谁评的 + 样本量" |
| `output/eval/<tag>_summary.json` | 机读摘要，阶段门和 `--compare` 读它 |
| `output/judge_budget.json` | 裁判调用累计花费 |
| `output/judge_score_cache.jsonl` | 打分缓存，续跑不重烧 |
| `ckpts/<stage>-lora/` | 各阶段的 LoRA adapter |
| `models/<stage>-merged/` | 各阶段合并后的全量模型，每个约等于一份基座大小 |
| `logs/pipeline/state.json` | 当前进度 |
| `logs/pipeline/<step>.log` | 每一步的完整输出 |

这些目录全部在 `.gitignore` 里，不入库。

---

## 十一、注意事项

**磁盘。** 每个阶段合并出来的全量模型都是一份完整基座大小，主线一共五个。
下一阶段只依赖上一阶段，跑到后面可以把更早的删掉。

**GPU 互斥。** 采样和评测需要推理服务占着卡，训练和合并需要卡是空的。编排器会自动起停，
但**不要手工另起一个推理服务**——编排器停不掉它，下一步训练就会 OOM。

**在线 GRPO 的显存。** `train.grpo` 里那五项（`vllm_gpu_util`、`move_model_batches`、
`sleep_level`、`offload_model`、`offload_optimizer`）是 colocate 模式下防 OOM 的一组约束，
任何一项放松都可能在几十步之后炸。调之前先想清楚为什么。

**裁判服务限流。** 判分并发默认压得很低（`judge.call_workers`），因为这类服务容易 429，
并发一高就变成"一起退避、一起重试"，反而更慢。在线 GRPO 里还额外加了一把跨进程文件锁。

**跑之前必须改的**：`configs/train.yaml` 的 `paths` 和 `model` 两段全是占位符，
不是任何一台真机上的路径。

**这不是 clone 即可复跑的工件。** 它需要一个 32B 规模的领域基座、多卡机器、
一份领域语料，以及一个可用的裁判模型服务。仓库里提供的是完整实现和可读的口径，
`examples/` 下的数据是合成的。

---

## 十二、验证

```bash
python -m compileall -q src scripts tests    # 语法
bash -n scripts/*.sh scripts/swift/*.sh      # shell 语法
python -m pytest                             # 离线契约测试，纯 CPU
python -m ruff check .                       # 代码规范
```

测试全是**离线用例**：手写输入、手写期望，不连任何服务，不代表模型真实表现。
它们保的是判定口径不许被无意改掉——规则层和奖励函数同时是训练硬门和评测指标，
口径一漂，训练信号和成绩单会一起错，而且错得一致，看不出来。

---

## 十三、逐段讲解

[docs/RESULTS.md](docs/RESULTS.md) 是全部数字的唯一出处：五阶段三件套、真涨门判读、
数据流追溯（每一格从多少条筛到多少条）、裁判标定的六档表和可分性矩阵。

[docs/interview_guide.md](docs/interview_guide.md) 把每一步从零讲一遍：
数据长什么样、模型怎么加载、rollout 怎么采、奖励怎么算、优势从哪来、
损失怎么落到每个 token、显存怎么省下来、评测怎么读。
