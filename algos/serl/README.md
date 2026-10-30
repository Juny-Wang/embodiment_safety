# SERL learner

当前实现：图像 DrQ-SAC，三路编码器，10个Q网络中抽取2个，gamma=.99。
actor含熵正则与自动temperature；当前backup_entropy=False，保持既有配置。
1:1示范/在线均匀抽样；没有把阶段真值作为额外策略输入。
learner.py仅依赖数据集接口和replay，不导入Task1奖励。
actor_worker.py负责推理/采样与可选在线更新，任务装配由runners完成。

vendor/serl_launcher保留当前实际使用的SERL实现，许可证见vendor/LICENSE。
依赖版本见dependencies.json。大数据与checkpoint不进入Git。
