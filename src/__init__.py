"""32B 强化学习训练链路的源码包。

五个子包对应流程里的五段：

- :mod:`src.data`       数据构建：重产、切分、建池、造四个阶段各自的训练数据
- :mod:`src.models`     外部模型客户端：推理服务、裁判服务
- :mod:`src.rewards`    判定层：确定性规则、裁判打分、标定表、在线奖励函数
- :mod:`src.training`   训练编排：把四个阶段串成一条线
- :mod:`src.evaluation` 评测：推理 + 三件套判分

另有 :mod:`src.config` 读配置，:mod:`src.utils` 放日志和并发这类横切件。
"""
