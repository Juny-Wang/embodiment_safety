# 目录职责

| 目录 | 职责 |
|---|---|
| tasks/task1_dimm_insert/env | 观测、动作、阶段奖励、终止、100 ms 边界 |
| tasks/task1_dimm_insert/teleop | 人工数采、三路相机、安全反馈与控制器 |
| tasks/task1_dimm_insert/scene | collection.usda 入口、接触补丁、依赖哈希 |
| tasks/task1_dimm_insert/configs | Task1 数采配置、数据清单 |
| algos/serl | DrQ-SAC、replay、优化器、仓库内 SERL 依赖 |
| runners | 数据准备、检查、训练、回合运行、评估 CLI |
| configs/experiments | 实验组合配置 |
| tests | 契约与奖励回归测试 |
| outputs | 不进入 Git 的数据索引、replay、日志和 checkpoint |

Task1 只保留当前入口，不保留旧 FakeEnv、旧 Task1 PPO 环境和旧低维 SAC 适配器。
其他任务和算法预留目录不属于此次 Task1 清理范围。
