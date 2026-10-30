# Franka 手动遥操作与实时安全反馈 v002

所有文件在 isaac-b:/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/。

## 使用
1. 在 Isaac Sim 5.0 中打开本目录 manual_franka_scene.usda（完整候选场景，主场景未改）。
2. 暂停时打开 Window > Script Editor，运行：
```python
exec(compile(open("/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/launch_panel.py").read(), "launch_panel.py", "exec"))
```
3. Play，等待两个物理帧，然后点击 Initialize from MEASURED pose。初始化不发送运动命令；必须通过URDF与真实刚体panda_hand的FK一致性检查。
4. 勾选理解演示阈值，选择当前任务步骤，点击 ARM。
5. XYZ按钮每次请求0.25mm世界轴移动，旋转按钮每次请求0.5度世界轴旋转；实际目标推进<=1mm/s、2度/s，另有关节目标变化率上限。末端参考是panda_hand，不是指尖接触点。
6. Gripper OPEN/CLOSE 驱动夹爪；抓取许可区、夹持力阈值尚未标定，显示灰色，不代表抓稳或安全。禁止把这一版当作已通过抓取验收。
7. STOP取消目标推进并暂停时间线；红色超限锁存，清除实际危险、Play后手动ARM才能恢复。Stop后必须重新Initialize。修改任务步骤会解除ARM。
8. 关闭监测面板会停止控制。日志记录在 runs/<时间>/。

## 配置与记录
safety_config.json是单一配置，SI单位；显示角度/高度差显式转换deg/mm。阈值是演示/待验证原型值，不是打印材料损伤极限。未知阈值为null，灰色。
task_frames.jsonl每个物理帧记录测量、颜色、停止原因、实测关节位置和关节驱动目标；config_snapshot.json记录配置版本。逐点contacts.jsonl记录外部实际接触；socket内部接触只记录最大求解重叠，避免重复轴承接触淹没日志。接触力来自法向冲量/对应dt，不含摩擦冲量；不能冒充腕部六维传感器或驱动力矩。
SafetyEngine是独立Python模块，后续RL可复用evaluate和observation；本轮不改训练代码、奖励或PPO。

## 边界
- 主场景、源USD、DIMM尺寸、卡扣几何、PPO及baselines未改。新候选引用本地Franka缓存；机器人home只在新候选中author初始reset状态及匹配驱动目标，不在运行时瞬移。
- 新候选物理步长1000Hz用于诊断，不宣称主场景GUI已有此设置。
- 图形只使用debug draw，不逐帧改原网格/材质或创建碰撞。
- 真实接触与正常关节驱动控制；没有写卡扣角度、没有入座后切换闭锁drive。
- 16度是当前几何干涉前的诊断停止线（约17度已相交），不是厂家开角或机械验收。
- 目前是手动遥操作，不含自动七步轨迹、自动放手许可或闭锁成功判定。松夹由用户手动决定。
- 无材料变形/损伤模型，因此不制造弯曲/断裂动画。
- 桌面安装、完整卡扣开闭和DIMM接触闭锁仍须独立机械验收。
\n控制初始化会读取实际驱动目标和实测关节差，保留小范围静态保持偏置，避免启用控制时撤掉原有保持作用；这不是完整重力模型。若正在运动或目标差过大，拒绝初始化。实际位置与速度始终单独测量，不能用目标速度代替实际速度。\n
局部IK有迭代次数上限，使用Lula的URDF正运动学模型校核，不在GUI物理回调中做无上限全局IK搜索。已验证的静止目标解会复用，但每次仍检查几何残差。此处数值求解容差不代表实物或仿真接触精度。

插入阻力只统计DIMM与环境的接触，正常指夹接触单独显示，不计入横向插入阻力。界面弯矩是环境法向接触对DIMM刚体原点的合矩，不是PCB结构弯曲应力。除安装基座panda_link0外，移动机器人与非当前步骤允许对象的接触超过0.05N会作为演示接触检测阈值停止；允许对象角色不代表允许接触面的几何标定已经完成。
目前仍能看到原SDF接触材质面索引警告；法向接触读取通过不代表摩擦材质或机械闭环验收通过。
\n超限行保持红色并显示记录峰值，即使短峰已消失；危险清除后需人工ARM确认。actual_force_colors.csv/color_acceptance.json是标准盒体真实加载的颜色链路验证，不是DIMM破损或装配成功演示。\n