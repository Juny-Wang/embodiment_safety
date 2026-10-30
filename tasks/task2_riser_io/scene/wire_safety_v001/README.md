# Task2 单线安全诊断 v001

本目录为隔离诊断与打印候选工作包。不是完整 Task2、不是强化学习环境，也不是材料强度认证。
主入口 scene/riser_io_preview.usda 未改动；原有 Franka、PPO、baselines 均未修改。

## 当前能看什么
- L0_sensor_bench.usda：固定连接座—三维刚体线链—末端外力的测量正对照。
- L1_one_guide.usda / L2_two_guides.usda：增加真实三角网格导柱的隔离候选。
- fixture_assembly.usda：打印工装候选展示；分体夹仅为定位夹候选。
- force_comparison.png：实际 PhysX 低载/高载曲线。
- measured_comparison.gif / measured_comparison.png：根据有效 PhysX 位置和载荷记录制作的二维回放，不是 Kit 录屏，不是材料变形或断裂动画。
- printable_fixture.png：直接由 STL 渲染的工装图。

先查看 reports/acceptance.json。guide_contact_diagnostic_ready=false 时，L1/L2 不能作为通过验收的绕柱任务。
即使该项通过，也只代表接触诊断，不代表按指定侧绕过两柱、端子定位和紧固成功。

## Isaac 中查看
1. 停止当前仿真，不保存无关场景的运行状态。
2. 打开本目录 L0_sensor_bench.usda。先以 L0 为入口。
3. Script Editor 执行：
   exec(compile(open("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_safety_v001/scripts/launch_monitor.py").read(), "launch_monitor.py", "exec"))
4. Play。监测面板显示连接关节力、最小曲率半径、导柱接触法向力、标准盒体正对照。
5. 标准盒体持续测量通过前，风险状态为灰色；未知量不填零。
6. 该面板不发送 Franka 指令、不抢占键盘、只警告；STOP MONITOR 仅关闭监测，不替代 Isaac 停止按钮。
本轮没有接管用户 GUI。GUI 手动操作验收状态以 acceptance.json 为准。

## 信号的含义
- Connector joint N：隔离连接座与第一段线之间 CableAttachment 的真实 PhysX 关节反力模长。
- 力矩在同一个子关节坐标系读取；不将力与力矩混为一个值。
- 这不是原 TE 实物内部应力，也不等于原主场景连接器已建立传感器。
- 导柱力来自接触张量；力箭头位于接触点，显示法向力。未验证切向摩擦力，不将法向力冒充合力。
- 最大箭头长度 50 mm；比例 25 mm/N，超出显示截断，数值不截断。
- 曲率来自离散中心线，是分段模型的几何指标。
- 1 N、35 mm 为演示阈值，不是厂家允许载荷或弯曲半径。
- 真实材料应力、弹性伸长、疲劳、断裂均未实现/未标定。
- 数值发散记录不得当作过载损伤。本包保留失败原始记录，最终对照图只使用通过测量检查的数据。

## 物理模型和台架
- 米制，重力 9.81 m/s²。
- 24 个直径 7 mm、总中心线长 460 mm 的刚体段，总线段质量 24 g。
- 球关节支持三维转动；相邻重叠胶囊明确过滤碰撞，非相邻段保留自碰撞。
- 固定连接座属于试验边界；CableAttachment 具有可读反力。
- 最终数值配置：960 Hz，TGS，articulation 位置/速度迭代 64/16。不是材料标定参数。
- 明确绑定摩擦材料：静摩擦 0.4、动摩擦 0.3、恢复系数 0；均为待标定值。
- 标准盒体 0.1 kg，理论重力 0.981 N；保持唤醒避免休眠读数被误认为零载荷。
- 恒力组为外力诊断，不是 Franka 真实夹持动作。
- --controlled-pull 为限力虚拟试验执行器：目标速度 10 mm/s，PD 力反馈，0.25 N 合力上限；没有直接写线段 pose 或 joint angle。
- 数值异常检测停止的是隔离测试，不是材料损伤判据。不得将无效数据补成正常曲线。

## 重跑（全部在 isaac-b）
cd /mnt/data2/wjy_SERL/franka_sim
/mnt/data2/isaacsim-5.0.0/python.sh -B ../Embodied_Safety/tasks/task2_riser_io/scene/wire_safety_v001/scripts/run_bench.py --scene L0_sensor_bench --force 0.25
/mnt/data2/isaacsim-5.0.0/python.sh -B ../Embodied_Safety/tasks/task2_riser_io/scene/wire_safety_v001/scripts/run_bench.py --scene L0_sensor_bench --force 1.5
/mnt/data2/isaacsim-5.0.0/python.sh -B ../Embodied_Safety/tasks/task2_riser_io/scene/wire_safety_v001/scripts/run_bench.py --scene L1_one_guide --force 0.25 --direction y --controlled-pull
运行会新建带时间戳的 runs 子目录，不覆盖原始结果。不加载主场景。

## FDM 候选清单（STL 单位 mm，按 100% 比例）
- mount_tile_160x120.stl：160×120×6 mm；孔阵列与导柱/夹座安装孔配合。
- guide_column_D36_H35.stl：柱直径 36 mm；柱体高约 35 mm，带 5 mm 底脚及圆滑顶部；需要两件用于双柱候选。
- relief_collar_lower_candidate.stl / relief_collar_upper_candidate.stl：76×44 mm 分体夹；孔道 7.4 mm。
- screw_custom_D10_P2.stl：自定义 D10、螺距 2 mm，18 mm 轴段，非 ISO M10。
- screw_custom_D10_P2_shaft36.stl：同牙型、36 mm 轴段，供厚夹座候选装配。
- nut_custom_D10_P2_clearance_030.stl：相配打印螺母，径向间隙 0.30 mm。
导柱每件两套短螺丝/螺母，分体夹两套长螺丝/螺母。打印间隙和夹座装配未实测，先试印。
底板需要实体支撑并给底部螺母留空间；本版尚未完成整套工装的桌边固定结构。

## 应力释放夹为什么尚未验收
7.4 mm 孔道相对 7 mm 线缆有 0.2 mm 径向间隙，没有已验证的压紧预载。
本轮不能声称它靠摩擦承担轴向拉力。后续必须明确可打印压紧机构/弹性衬垫及材料，
通过有夹/无夹、滑移与连接载荷对照验证。不得用固定线段或切换约束冒充应力释放。

## 线的变体与难度边界
已生成 L0/L1/L2 导柱数量变体；不是完整可训练布线课程。
线长、直径、刚度、摩擦的训练随机化保持关闭，原因是材料和可达路径尚未标定。
后续先形成独立标定表，再生成短/标准/长线和软/中/硬线，逐一验证长度余量、最小半径和夹爪通道。
四线交叉、近端损伤等效演示、自动成功判定没有接入主场景。

## 主场景审计阻塞项
reports/source_audit.json 含实际 prim、变换、关节配置、锚点误差和源文件 SHA256。
四个环端子 FixedJoint 的旋转四元数未归一化（长度约 0.5）。
原线链最大锚点误差约 1.09–1.83 mm，且只有局部 Z 轴转动自由度。
本轮没有通过修改主场景隐藏这些问题；必须先完成原模型关节修订与回归，再接入新工装。
