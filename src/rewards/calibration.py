"""裁判标定：先量尺子，再拿尺子去优化。

本模块在整条链路里的位置：选样门和评测判据的依据。冷启动、拒绝采样、偏好对三处都要回答
同一个问题——"这条 think 真的比那条干净吗，还是只是这一次判分抖上去了"。本模块提供的
就是回答它所需要的两个量。

**为什么必须先标定。** 一个更省事的做法是：让裁判各打一次分，谁高选谁。问题在于裁判
本身有噪声，同一段 think 打两遍分数会差。干净这一端的方差尤其大：越接近"完全没照抄"，
两段质量相仿的 think 打出来能差一两分。不标定就选样，选进训练集的会有相当一部分只是
"这一次抖高了"的样本；训完看指标涨了一点，也分不清是真涨还是评测那次抖高了。

**标定怎么做的。** 取 13 个参考问答对，每个人工做出 6 档已知脏度的 think——
"抄 0 句 / 1 句 / 2 句 / 3 句 / 4 句 / 整段抄"，共 13×6 = 78 条；让裁判对每一条各打 16 遍，
合计 1248 次判分调用。16 遍的均值就是该档的分数，16 遍的标准差就是该档的档内噪声 σ。
结果写在 `configs/train.yaml` 的 `calibration.tiers`。

**量出来的三条事实，直接决定了整条链路的数据造法：**

1. **整条梯子被系统性压低。** 连"完全没照抄"裁判也只给 7.34 分，不给满分。
   但单调性还在——Spearman 等级相关 0.954，排序方向是对的。所以它能当**相对**尺子用，
   不能当**绝对**分数读。
2. **越靠干净端 σ 越大。** "没抄"档 σ≈1.85、"抄 1 句"档 σ≈1.63，而"抄 4 句"档 σ≈0.06。
   6 个相邻档里只有"抄 3 句↔抄 4 句"那一对过得了 3σ 可分判据；"没抄↔抄 1 句"的
   σ 距离只有 0.8，完全分不开。→ **偏好对只能造"基本干净 vs 重度照抄"这一种，
   中间档一律不用。**
3. **多打压不掉它。** "没抄"档从 k=1 打到 k=16，观测标准差只从 2.13 降到 1.85。
   卡死的是样本间方差不是单遍噪声，加 k 救不了。→ **整体评测靠加题数 N，不靠加 k。**

完整的可分性矩阵和各档 k 曲线见 `docs/RESULTS.md` 第 5 节。

**这张表挡不住什么。** 它量的是裁判在"照抄多少"这一个维度上的**方差**。裁判如果对某类
题材系统性偏高或偏低，表里看不出来——那是**偏差**，得靠换一批标定样本才测得到。
"""

from __future__ import annotations

from src.config import Config, load_config

# ---------------------------------------------------------------------------
# 标定结果。默认值从 configs/train.yaml 的 calibration 段读，这里只是缓存。
# ---------------------------------------------------------------------------
_cfg: Config | None = None


def _calib() -> Config:
    """取配置里的 calibration 段，进程内只解析一次。"""
    global _cfg
    if _cfg is None:
        _cfg = load_config().calibration
    return _cfg


def calibration_tiers() -> list[tuple[float, float]]:
    """六档标定表：``[(该档分数均值, 该档档内标准差), ...]``，由脏到干净。

    :return: 六个二元组
    """
    return [(float(mean), float(sigma)) for mean, sigma in _calib()["tiers"]]


def sigma_of_score(score: float) -> float:
    """查一个分数落在哪一档，给出该档的标准差。

    档与档之间做线性插值：真实的 think 分数是连续的，不会正好落在六个锚点上。
    分数超出表的两端就取端点值，不外推——外推出来的 σ 没有任何测量支撑。

    :param score: 裁判给的干净分（0~10）
    :return: 该分数处的标准差
    """
    tiers = calibration_tiers()
    clamped = max(tiers[0][0], min(tiers[-1][0], float(score)))
    for (x0, y0), (x1, y1) in zip(tiers, tiers[1:], strict=False):   # 相邻两档配对，最后一档没有下一档
        if x0 <= clamped <= x1:
            if x1 == x0:
                return y1
            ratio = (clamped - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return tiers[-1][1]


def confident_cleaner(score_clean: float, score_dirty: float, n_sigma: float = 2.0) -> bool:
    """判 `score_clean` 这条是不是**按 N 倍标准差带不相交**地比 `score_dirty` 那条更干净。

    判据是 ``差值 > N × (σ_clean + σ_dirty)``，两个 σ 各自查表。意思是：两条 think 的分数
    各自带一条误差带，只有两条带完全错开，才承认"确实更干净"，否则算读不出来、这条样本不要。

    这是整条链路的选样门。冷启动选改写稿、拒绝采样选候选、偏好对选 chosen，全走它。

    :param score_clean: 候选（希望更干净的那条）的分数
    :param score_dirty: 对照（基座原推理）的分数
    :param n_sigma: 要求带宽的倍数。2 是主线，调到 3 更严但产出率会掉很多
    :return: 可分为 True
    """
    gap = float(score_clean) - float(score_dirty)
    if gap <= 0:                                       # 连均值都没更高，不必再看误差带
        return False
    return gap > n_sigma * (sigma_of_score(score_clean) + sigma_of_score(score_dirty))


def eval_se(n_items: int, k: int) -> float:
    """整体评测的标准误：N 道题、每题打 k 遍，模型平均干净分的抖动有多大。

    公式 ``SE = √(σ_between² + σ_judge²/k) / √N``。两项的分工是关键：

    - ``σ_judge`` 是裁判自己的手抖，同一段 think 反复打分的差异。打 k 遍能压到 ``σ_judge/√k``。
    - ``σ_between`` 是题与题之间的真实差异，有的题就是好写、有的题就是难写。**k 压不掉它**，
      只能靠加题数。

    直接后果：k 从 3 加到 16，SE 几乎不动，钱白花。要把判据收紧只能加 N。

    代入本项目的实测代表值（σ_judge=0.74、σ_between=1.02）：
    **N=500、k=3 时 SE ≈ 0.0495，真涨门 3×SE ≈ 0.15。** 主线验收就用这一档，
    而不是把每条打 16 遍——大 N 小 k 性价比最高。

    :param n_items: 评测题数
    :param k: 每题打分遍数
    :return: 平均分的标准误
    """
    calib = _calib()
    sigma_judge = float(calib["sigma_judge"])
    sigma_between = float(calib["sigma_between"])
    sigma_total = (sigma_between ** 2 + sigma_judge ** 2 / max(1, k)) ** 0.5
    return sigma_total / max(1, n_items) ** 0.5


def true_gain_threshold(n_items: int, k: int, n_sigma: float = 3.0) -> float:
    """两个阶段的平均分差要大过多少，才算"真涨"而不是抖动。

    取 ``n_sigma × SE``。差值小于它就是读不出来，两个阶段按统计打平记，不许说谁更好。

    :param n_items: 评测题数
    :param k: 每题打分遍数
    :param n_sigma: 几倍标准误算真涨
    :return: 真涨门限（分）
    """
    return n_sigma * eval_se(n_items, k)
