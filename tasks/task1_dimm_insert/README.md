# Task1：当前 SERL 训练交付

所有命令在远端执行：
```bash
ssh isaac-b "cd /mnt/data2/wjy_SERL/Embodied_Safety && ..."
```

## 当前接口

- 场景：`tasks/task1_dimm_insert/scene/collection.usda`。与已确认的采集场景保持相同内容，底层 USD/贴图为远端只读引用。
- 策略：external、wrist_left、wrist_right 三路 uint8 RGB，每路 256×256，T=1。
- 状态 31D：9 关节位置 + 9 关节速度 + 实测夹爪宽度 + 3 目标位置 + 4 目标四元数 + 夹爪目标，共 27D，再加 4 个阶段已获奖标记。
- 不把 socket/DIMM 真值、卡扣角度、接触力、刚度直接送入策略；这些用于环境奖励及诊断。
- 动作 7D，范围 [-1,1]：末端 xyz 线速度、xyz 角速度、夹爪速度。
- 单位缩放：0.32 m/s、1.57079632679 rad/s、0.06 m/s；最大目标超前 5 mm。不是关节速度“用米”。
- 每动作保持 100 ms，对应 24 个 240 Hz 物理步。中断不补齐、不补造下一观测。
- 真终止 mask=0，时间截断 mask=1；传感失效不作为有效训练样本。
- 每回合清空奖励标记、终止历史与启动辅助释放历史。

## 奖励

实现：`env/reward.py`；条件：`env/milestones.py`；终止：`env/termination.py`。
奖励版本 `task1_stage_once_v001`，接口 `task1_rgb_state31_reward_history_v001`。

| 阶段 | 每回合最多一次 |
|---|---:|
| 两侧开扣且撤离后路径可用 | 0.05 |
| 许可区抓稳并脱离置物架 | 0.10 |
| 插入准备位与姿态对正 | 0.05 |
| 入座与双侧锁止 | 0.20 |
| 松夹撤离并保持 | 1.00 |

最大总和 1.40。这是一次性阶段奖励，不是势函数塑形。
当前 0.20 入座奖励保守地在最终验收边界发放。
终局使用已接受场景上的操作判据：开到闭历史、落座、支撑接触、启动辅助释放、
松夹、撤离与稳定保持；不是额外的材料强度认证。
诊断力阈值：两侧按钮各3N，DIMM环境5N，两指各5N；不称厂家额定值。
启动闭位辅助在开扣接触/转动后单向释放，不在入座时重新开启；关闭 XY 吸附。

## 实际训练数据

`configs/data_manifest.json` 给出本机路径。
- 当前可直接训练：1个完整回合的1349条转移，加493条严格对齐的历史片段，共1842条。完整回合回报1.40。
- 原始1350条中的最后1条已位于因果成功终止之后，索引不纳入，原文件不改。
- 迁入28个有效在线诊断转移，不含人为注入急停样本。
- 历史15条标签已迁入 outputs/datasets/task1/aligned_history/historical_labels。03–15去重；01/02/03分别选入163/163/167条。\n- 50ms相邻行或100ms内子步只有动作一致、24条指令连续、位置/旋转/夹爪目标逐步积分一致、RGB有效时才接入；其余不平均、不插值伪造。\n- 完整回合因果计算阶段标记，再选取片段；不在片段开头清空奖励历史。片段不冒充新增完整成功回合。
- 没有独立留出完整评估回合；不能声称“15条独立示范已正式训练”。
- 大图像和原始记录仍位于远端 franka_sim，只读引用，文件哈希在输出 manifest 内。

## 运行

离线检查解释器：
`/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/.print_venv/bin/python`。
学习器解释器：
`/home/ufdwaa/anaconda3/envs/env_isaaclab/bin/python`。
Isaac：`/mnt/data2/isaacsim-5.0.0/python.sh`。
外部目录只提供已有第三方依赖，SERL 项目源码使用本仓库 algos/serl/vendor。

在仓库根目录执行：
```bash
# 数据准备：仅首次，输出已存在时拒绝覆盖
<离线Python> -B -m runners.prepare_task1
<离线Python> -B -m runners.preflight
<离线Python> -B -m unittest discover -s tests

# 固定数据混合更新；默认1:1示范/在线，batch=2、UTD=1，为小预算起始配置
<学习器Python> -B -m runners.train --updates 100 --output outputs/runs/task1/serl/first
# 续训：恢复优化器、参数和采样器状态
<学习器Python> -B -m runners.train --updates 100 --checkpoint outputs/runs/task1/serl/first/checkpoint.msgpack --output outputs/runs/task1/serl/continued

# 短回合接口验证（不是任务成功评估；动作限为2mm/s，旋转和夹爪禁用）
python3 -B -m runners.rollout --smoke --checkpoint <checkpoint.msgpack> --output outputs/runs/task1/serl/smoke

# 完整原生时钟回合；完整7D动作、真实接触，--learn每10个动作做混合更新
python3 -B -m runners.rollout --learn --checkpoint <checkpoint.msgpack> --output outputs/runs/task1/serl/episode
# 独立评估：确定性策略、不开学习、数据不混进训练replay
python3 -B -m runners.eval --checkpoint <checkpoint.msgpack> --output outputs/runs/task1/serl/eval
```

每个输出目录必须为新目录，避免覆盖。完整回合最多240秒仿真时间。
每次命令启动新的Isaac进程，关闭后下一次从原始场景复位。
人工启动：Isaac打开 scene/collection.usda，Script Editor运行 teleop/launch_collect.py。
不要同时在同一进程载入旧 franka_sim 包与新仓库包。

固定数据更新成功不等于完整策略任务成功；查看 configs/validation.json 的实际测试范围。

历史对齐回归已通过22项测试，并在 outputs/runs/task1/serl/history_aligned_check 完成2次实际混合更新；日志记录抽到02和03的历史片段。新数据索引有独立哈希，旧索引的checkpoint不会被静默续训到新数据集。

当前回归测试共27项。控制器停止时保存停止前IK残差、候选关节限位余量和目标/实测位姿；真实失败末段按已观测物理步写入终止转移（mask=0），不把已发出但尚未执行的命令计入100ms边界。验证结果见 outputs/runs/task1/serl/aligned_budget100_eval_diagnosed/diagnosis.json；该次策略未完成任务，不代表训练已收敛。
