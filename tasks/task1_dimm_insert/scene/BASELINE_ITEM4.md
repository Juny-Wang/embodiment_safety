# 基线采集 · 第 4 项（BASELINE）— 基础设施 + 协议（部分完成）

日期：2026-10-23（Asia/Shanghai）  
状态：**基础设施就绪；阈值 TBD，待用户完成真实跑次后才写入 `safety_config` / `geometry_calib`。**  
场景：`franka_manual_safety_v002`（含 Item3 `insertion_clearance`）

依据：`CLEARANCE_MERGE_ITEM3.md` §5；本目录 `baseline_logger.py` + `keyboard_panel.py` 挂钩。

---

## 0. 硬性约定

| 项 | 约定 |
|----|------|
| 阈值 | **TBD** — 未完成 N 次成功跑前 **禁止** 改 `insertion_clearance.*` / 旧 rule limit |
| 真机 | 禁止改真实 Franka |
| 备份 | 改 config 前必须备份 |
| `end_difference_mm` | 姿态代理，非真实两端入座差；对照完成前勿对外当「真实入座」宣传 |
| 日志默认 | **关闭**；`BASELINE_LOG` 未开时 teleop 行为不变 |

---

## 1. 如何开启日志（推荐）

### 方式 A — 环境变量（推荐，启动前）

在启动 Isaac / 跑 `keyboard_panel.py` **之前**：

```bash
export BASELINE_LOG=1
# 可选：
# export BASELINE_LOG_DIR=/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/baseline_logs
# export BASELINE_LOG_PATH=/path/to/custom_run.csv
# export BASELINE_TRIAL_TAG=success_01
```

然后照常 Script Editor 执行 `keyboard_panel.py`。  
CSV 默认：`{场景目录}/baseline_logs/baseline_YYYYMMDD_HHMMSS.csv`

### 方式 B — 面板已起来后（Script Editor）

```python
exec(open("/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/baseline_logger.py").read())
# 若主面板绑定为 _franka_keyboard：
attach(_franka_keyboard, enable=True, trial_tag="success_01")
# 换试次标签（同一 CSV 内区分）：
_franka_keyboard.set_baseline_trial_tag("success_02")
# 关闭：
_franka_keyboard.disable_baseline_log()
```

### 关闭

```bash
unset BASELINE_LOG
# 或 BASELINE_LOG=0
```

禁用时 `maybe_log` 为空操作，不影响键盘遥操作。

### CSV 字段

见 `baseline_logger.CSV_FIELDS`（含 Item3 §5 所列：`align_xy_m`, `tilt_deg`, `tilt_t1/t2`, `end_difference_mm`, `seated_depth_m`, `insertion_clearance_*`, 力、`step_name`, `block_down`, 卡扣角等）。

---

## 2. 跑前自检清单

- [ ] 场景为 `franka_manual_safety_v002`，Plan D session 已加载（弱卡扣 / 细接触）
- [ ] 确认 metrics：`socket_frame_used == /World/MatchedSocket/Asset`
- [ ] 确认存在 `insertion_clearance_ratio` / `_ok` / `_color`
- [ ] `BASELINE_LOG=1` 已开；CSV 路径可写；开跑后行数在增加
- [ ] 每试次设置 `BASELINE_TRIAL_TAG` 或 `set_baseline_trial_tag(...)`
- [ ] **不**在本阶段改 `safety_config.json` 阈值数字

### Item3 静态复核（基础设施阶段，非 live stage）

| 检查 | 期望 | 结果（填） |
|------|------|------------|
| `plan_d_runtime` 使用 Asset 帧 | `socket_asset_prim` / paths.socket | ✅ 远端静态 OK |
| `evaluate_insertion_clearance` 存在 | 函数可 import | ✅ |
| `safety_config.insertion_clearance` | 段存在，demo 数未改 | ✅ |
| `baseline_logger` + panel `maybe_log` | 已同步 | ✅ |

> Isaac MCP / 本盒子直连 live stage：**不可用**（无 Isaac MCP；盒子→isaac-b SSH 超时）。
> 远端文件静态复核 2026-10-23 19:11 Asia/Shanghai：**ALL_OK**。
> live 姿态复核需用户在 Isaac 内看 panel metrics 或 CSV 列 `socket_frame_used`（本阶段未要求用户 teleop）。

---

## 3. 协议（自 Item3 §5）

1. **场景**：v002 + Item3 代码；确认 Asset 帧。  
2. **日志**：每物理步 CSV（见上）。  
3. **成功 × ≥5**：GRASP→ALIGN→INSERT→LATCH→RELEASE；无硬 RED；最终 `dual_latched`。分析窗：  
   - W1 = 首次 DIMM–Socket **环境**接触前 0.2 s  
   - W2 = 入座稳定后（建议 dual_latched 后 0.5 s）  
4. **故意失败 × 每类 ≥3**：纯 XY / 纯 tilt / 纯一端翘 / 过深猛插；核对 `insertion_clearance_worst` 与 `block_down`。  
5. **对照**：接触点沿 insert 的两端 Z 差 vs `end_difference_mm`。  
6. **阈值建议（仅分析后填写，本文件先留空）**：stop≈success P95×余量 或 failure P5；warn≈0.8×stop。  
7. **卡扣**：open≈15° 意图 vs 运行时≈40°；`closed_angle_abs_deg_max` + seated。  
8. **产出**：填本表 + CSV/NPZ 路径；**再**改 config（另开变更，先备份）。  
9. **禁止**：真机；无备份改阈值；未对照宣称 end=真实入座。

---

## 4. 成功跑次表（≥5）— 结果待填

| # | trial_tag | CSV 路径 | 最终 dual_latched | 曾 RED? | W1 max xy_mm | W1 max tilt_deg | W1 max end_mm | W1 max ic_ratio | W2 seated_depth_mm | 备注 |
|---|-----------|----------|-------------------|---------|--------------|-----------------|---------------|-----------------|--------------------|------|
| 1 | | | □ | □ | | | | | | |
| 2 | | | □ | □ | | | | | | |
| 3 | | | □ | □ | | | | | | |
| 4 | | | □ | □ | | | | | | |
| 5 | | | □ | □ | | | | | | |
| 6+ | | | | | | | | | | |

成功汇总（跑完后填）：

| 分量 | N | P50 | P90 | P95 | max | 建议 stop (TBD) | 建议 warn (TBD) |
|------|---|-----|-----|-----|-----|-----------------|-----------------|
| align_xy_m (mm) | | | | | | **TBD** | **TBD** |
| tilt_deg | | | | | | **TBD** | **TBD** |
| end_difference_mm | | | | | | **TBD** | **TBD** |
| insertion_clearance_ratio | | | | | | **TBD** (=1.0 归一) | **TBD** |
| seated_depth_m (mm) | | | | | | **TBD** | **TBD** |

---

## 5. 故意失败表（每类 ≥3）— 结果待填

### 5.1 纯 XY 偏

| # | trial_tag | CSV | worst 期望=xy | worst 实际 | block_down? | 备注 |
|---|-----------|-----|---------------|------------|------------|------|
| 1 | | | xy | | □ | |
| 2 | | | xy | | □ | |
| 3 | | | xy | | □ | |

### 5.2 纯 tilt

| # | trial_tag | CSV | worst 期望=tilt | worst 实际 | block_down? | 备注 |
|---|-----------|-----|-----------------|------------|------------|------|
| 1 | | | tilt | | □ | |
| 2 | | | tilt | | □ | |
| 3 | | | tilt | | □ | |

### 5.3 纯一端翘（end）

| # | trial_tag | CSV | worst 期望=end | worst 实际 | block_down? | 备注 |
|---|-----------|-----|----------------|------------|------------|------|
| 1 | | | end | | □ | |
| 2 | | | end | | □ | |
| 3 | | | end | | □ | |

### 5.4 过深猛插（力/轴向，非 clearance 主类）

| # | trial_tag | CSV | axial 峰值 N | 灯色 | 备注 |
|---|-----------|-----|--------------|------|------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

---

## 6. end_difference 对照（待填）

| trial_tag | 帧/时间 | end_difference_mm（姿态） | 接触点 ΔZ_mm（沿 insert） | 差 | 备注 |
|-----------|---------|---------------------------|---------------------------|----|------|
| | | | | | |
| | | | | | |

结论（待填）：□ 可继续用代理  □ 需改公式  □ 数据不足

---

## 7. 卡扣 / seated（待填）

| 项 | 测量 | 建议 warn | 建议 stop | 状态 |
|----|------|-----------|-----------|------|
| 开扣「够开」角 | | TBD | TBD | 待测 |
| 关节硬限位 | ~40°（Item1） | — | — | 已审计 |
| `closed_angle_abs_deg_max` | demo 3.0 | TBD | TBD | 未改 |
| `seated_depth_tol_m` | demo 1.5 mm | TBD | TBD | 未改 |

---

## 8. 阈值写入检查（**现在全部 TBD — 勿改 config**）

| 键 | 当前 demo | 新值 | 写入？ |
|----|-----------|------|--------|
| `insertion_clearance.xy_tol_m` | 0.0025 | TBD | □ 禁止直至跑完 |
| `insertion_clearance.tilt_tol_deg` | 2.0 | TBD | □ |
| `insertion_clearance.end_difference_tol_mm` | 1.0 | TBD | □ |
| `insertion_clearance.warning_ratio` | 0.8 | TBD | □ |
| 旧 `rules` align/tilt/end limits | 同上 | TBD 派生 | □ |

---

## 9. 文件与同步

| 路径 | 说明 |
|------|------|
| `/workspace/plan_d_work/out/baseline_logger.py` | 日志模块 |
| `/workspace/plan_d_work/out/keyboard_panel.py` | 可选挂钩 |
| `/workspace/plan_d_work/out/BASELINE_ITEM4.md` | 本协议 |
| `/workspace/plan_d_work/out/baseline_logs/` | 本地 smoke CSV |
| isaac-b：`.../franka_manual_safety_v002/baseline_logger.py` 等 | 场景目录副本 |
| 远端日志目录（建议） | `.../franka_manual_safety_v002/baseline_logs/` |

**本阶段未修改**：`safety_config.json` / `geometry_calib.json` 中的数值 limit。

---

## 10. 进度

- [x] 协议文档 + 空表  
- [x] `baseline_logger.py` + panel 可选挂钩  
- [ ] 用户完成 ≥5 成功 + 每类 ≥3 失败  
- [ ] 填表 + 阈值建议  
- [ ]（另变更）备份后写入 config  



---

## 11. 同步记录

- 本地 backup：`keyboard_panel.py.bak_item4_20261023_191012`
- 远端 backup：`.../franka_manual_safety_v002/.bak_item4_20261023_191125`
- 已推送：`baseline_logger.py`, `keyboard_panel.py`, `BASELINE_ITEM4.md`
- 远端已建：`baseline_logs/`
- Item3 静态 verify：ALL_OK（非 live stage）
