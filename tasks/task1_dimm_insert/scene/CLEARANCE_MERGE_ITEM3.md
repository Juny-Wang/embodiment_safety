# 插入间隙门禁合并 · 第 3 项（CLEARANCE_MERGE）

日期：2026-10-23（Asia/Shanghai）  
范围：把几何插入间隙相关度量合并为 **一条** 决策门；保留旧键兼容；**不发明新标定 mm**；Item4 再基线重调。  
训练场景：`franka_manual_safety_v002`

依据：`MEASUREMENT_SPEC_ITEM2.md`、`geometry_calib.json`、`step_fsm.py`、`safety_config.json`、`plan_d_runtime.py`、`keyboard_panel.py`

---

## 1. 当前重叠 / 冲突（合并前）

### 1.1 Socket 父 vs Asset 坐标系（最严重）

| 用途 | 帧 |
|------|-----|
| 基类 `measure`：插入轴、tilt、end_difference、力轴向 | `/World/MatchedSocket/Asset`（`SOCKET`） |
| Plan D `compute_alignment_metrics`（合并前） | `/World/MatchedSocket`（**父** Xform） |
| `safety_config.paths.socket` / 铰链 / 卡扣 | Asset |

`geometry_calib.socket_local.entry_point` / `seated_reference` 是按 **Asset 局部网格** 标定的。若父相对 Asset 有平移/旋转，`align_xy_m` / `seated_depth_m` 会相对真正插入腔漂，而 `tilt_deg`（基类）仍相对 Asset → **同一帧内 XY 与 tilt 说的不是同一套坐标**。

**Item3 修复**：`compute_alignment_metrics` 改为读 `frames.socket_asset_prim`（默认 Asset）。仅改姿态查询，**不改物理 / 碰撞**。

### 1.2 双 tilt 公式

| 源 | 公式 | 何时写入 `tilt_deg` |
|----|------|---------------------|
| **主**（基类 / Item2 §2.4） | `max( acos(\|n_board·Y_sock\|), asin(\|long·insert\|) )` | `measure()` 正常时 |
| **备用**（Plan D 旧） | `\|90° − acos(\|n_board·Z_sock\|)\|` | 旧键盘：仅当 `tilt_deg is None` 时 **静默覆盖** |

两式几何含义不同；备用混入门禁会导致「同姿态、不同 tilt」。

**Item3 修复**：

- 新增 `tilt_deg_primary`（与基类同式）+ `tilt_t1/t2`。
- 键盘：`tilt_deg is None` 时只填 **primary**，不再用 board-Z 作 `tilt_deg`。
- `tilt_board_to_socket_z_deg` **仅调试保留**，不进 gate。

### 1.3 `end_difference_mm` vs 真实入座深度

| 度量 | 含义 | 用于 |
|------|------|------|
| `end_difference_mm` | `\|long·insert\| × length × 1000` — **姿态两端高差代理**，不是接触点 Z 差 | 旧 traffic rule（contact_gated）；现并入 clearance |
| `seated_depth_m` | Asset 局部：`dimm_z − seated_reference_z` | `seated_ok` → **dual_latched / RELEASE** |
| `align_z_m` | 相对 `entry_point` 的 Z | 显示 / 进给参考，非 gate |

冲突点：

1. 旧 safety rule 把 `end_difference` / `tilt` 标成 **contact_gated**，而 ALIGN 阶段常无接触 → 灯不亮，但 `step_fsm.alignment_ok` 又用 **无门控** 的 `tilt_deg` + `align_xy` → **灯与 block_down 不同步**。
2. `seated_ok` 用 `\|depth\|≤1.5mm` **且** `dxy ≤ 2×xy_tol`（历史 2× 带宽），与 ALIGN 的 `xy_tol` 不一致；它属于 **锁扣门**，不应再塞进「插入间隙」。
3. `end_difference` **≠** 两端真实入座差（Item2 已标待 Item4 对照接触点）。合并后仍用同一代理公式，阈值仍为 demo **1.0 mm**。

### 1.4 多重独立门如何互相打架

合并前同时存在：

1. `step_fsm.alignment_ok`：`\|xy\|>xy_tol` **或** `\|tilt\|>tilt_tol` → `block_down`（ALIGN/INSERT）
2. TrafficLight：`align_xy_m`（键盘硬喂，limit=xy_tol）；另有 config 里 `tilt_deg`/`end_difference_mm`（contact_gated，常不生效）
3. `dual_latched`：角度 + `seated_ok`（含另一套 xy 带宽）

结果：可能 **block_down=True 但灯仍绿**（tilt 超限而 light 只看了 xy），或 **灯黄/红但原因键与 gate 字符串不一致**。

---

## 2. 单一间隙度量 Schema：`insertion_clearance`

### 2.1 分量（全部姿态、无接触门控）

```
xy_ratio   = |align_xy_m|        / xy_tol_m          # demo 0.0025 m
tilt_ratio = |tilt_deg|          / tilt_tol_deg      # demo 2.0 deg；tilt = primary
end_ratio  = |end_difference_mm| / end_tol_mm        # demo 1.0 mm；有值才计入
```

`seated_depth_m`：**只报告**（`insertion_clearance_seated_depth_m`），**不**进入 ALIGN/INSERT 的 ratio / `block_down`。

### 2.2 单一决策（红黄）

```
ratio = max( active component ratios )
       active = {xy, tilt} 必选；end 有值则加入

GREEN  : ratio < warning_ratio     (demo 0.8)
YELLOW : warning_ratio ≤ ratio < 1.0
RED    : ratio ≥ 1.0
GRAY   : xy 或 tilt 缺失
```

- `insertion_clearance_ok` ⇔ 非 RED（且非 GRAY）→ GREEN/YELLOW 都允许下行（黄=警告）
- `insertion_clearance_worst` = 最大分量名
- `insertion_clearance_reason` = `ok` / `clearance_{worst}_warn` / `clearance_{worst}_out` / `missing_…`

### 2.3 与 `block_down` 映射

```
ALIGN / INSERT:
  alignment_ok() 优先读 insertion_clearance_ok
    False → block_down = True，reason = gate_not_aligned:{insertion_clearance_reason}
  无 clearance 字段时回退旧 xy+tilt 逻辑（兼容）

LATCH_LOCK / RELEASE:
  仍用 dual_latched（卡扣角 + seated_ok）— 不与 clearance 混用
```

TrafficLight：主喂 `insertion_clearance`（limit=1.0）；`align_xy_m` 仅在 clearance 缺失时兼容喂入。旧 `tilt_deg`/`end_difference_mm`/`align_xy_m` **规则条目保留**（显示/日志），不删除。

### 2.4 输出字段（merge 进 metrics）

| 键 | 说明 |
|----|------|
| `insertion_clearance` / `_ratio` | 最差分量比 |
| `insertion_clearance_ok` | bool |
| `insertion_clearance_color` | GRAY/GREEN/YELLOW/RED |
| `insertion_clearance_worst` | xy\|tilt\|end |
| `insertion_clearance_reason` | 短因 |
| `insertion_clearance_*_ratio` | 各分量 |
| `insertion_clearance_tol` | 当前所用 tol 快照 |
| **兼容保留** | `align_xy_m`, `tilt_deg`, `end_difference_mm`, `seated_depth_m`, `seated_ok`, `tilt_board_to_socket_z_deg`(debug) |

---

## 3. 代码改动（非破坏草稿）

| 文件 | 改动 |
|------|------|
| `plan_d_runtime.py` | Asset 帧；`compute_tilt_primary_deg`；`evaluate_insertion_clearance`；对齐时同步算 `end_difference_mm` |
| `step_fsm.py` | `alignment_ok` 优先 `insertion_clearance_ok` |
| `keyboard_panel.py` | primary tilt / end 回填；调用 evaluate；灯喂 clearance |
| `safety_config.json` | 新增 `insertion_clearance` 段 + rule `insertion_clearance` limit=1.0；旧键保留 |
| `geometry_calib.json` | `socket_frame_for_align` → Asset；Item3 注释 |

备份：`*.bak_item3_20261023_190541`（同目录）

**未做**：删除旧 metric；改物理；改 demo 数值；跑全流程基线（Item4）。

---

## 4. Demo 数值（保持，Item4 重调）

| 参数 | 值 | 来源 |
|------|-----|------|
| `xy_tol_m` | 0.0025 | `geometry_calib.alignment` |
| `tilt_tol_deg` | 2.0 | 同上 |
| `end_difference_tol_mm` | 1.0 | 旧 `rules[end_difference_mm].limit` |
| `warning_ratio` | 0.8 | `safety_config.warning_ratio` |
| `seated_depth_tol_m` | 0.0015 | 仅 seated_ok，非 clearance |

---

## 5. Item4 应跑的基线协议（精确清单，本项不执行）

目标：用 **正常成功插入** 流标定 warn/stop，替换 demo 数。

1. **场景**：`franka_manual_safety_v002` + 当前 Plan D session（含 Item3 代码）。确认 `socket_frame_used == /World/MatchedSocket/Asset`。
2. **日志**：每物理步（或 ≥50 Hz）记录  
   `align_xy_m, tilt_deg, tilt_t1/t2, end_difference_mm, seated_depth_m, insertion_clearance_* , left/right_press_N, axial/lateral_contact_N, finger_* , step_name, block_down, latch angles`。
3. **成功轨迹 × N（建议 ≥5）**：人工或脚本走 GRASP→ALIGN→INSERT→LATCH→RELEASE，**无 RED、最终 dual_latched**。取各分量在「首次 DIMM–Socket 环境接触前 0.2 s」与「入座稳定后」的分布。
4. **故意失败 × 每类 ≥3**：纯 XY 偏、纯 tilt、纯一端翘、过深猛插；确认 `worst` 标签与 `block_down` 一致。
5. **对照**：同一批帧用两端接触点沿 insert 的 Z 差 vs `end_difference_mm`；报告相关/偏差（决定是否换公式）。
6. **阈值建议**：  
   - stop (tol) ≈ success P95 × 安全余量，或 failure P5 下界（取更严且可操作者）  
   - warn = 0.8 × stop（或 success P90）  
   - 写入 `insertion_clearance.*` **与** 同步派生旧键 limit（仍不删旧键）
7. **卡扣**：复测 Item1 的 open≈15° 意图 vs 运行时 ≈40°；`closed_angle_abs_deg_max` 与 seated 一并出 warn/stop。
8. **产出**：`BASELINE_ITEM4.md` + 更新后的 `safety_config.json` / `geometry_calib.json`；附原始 CSV/NPZ 路径。
9. **禁止**：改真机 Franka；无备份不写；不在未对照接触点前把 end 代理当「真实入座差」对外宣传。

---

## 6. 验收自检（Item3）

- [x] 文档写明父/Asset、双 tilt、end vs seated 冲突  
- [x] 单一 `insertion_clearance` + 红黄规则 + `block_down` 映射  
- [x] 旧键保留；demo 数未新造  
- [x] 对齐帧改 Asset；tilt 门禁只用 primary  
- [x] 备份已建  
- [x] isaac-b 同步 OK（经 Windows 中继 password；远端备份 `.bak_item3_20261023_190834`）
      路径：`.../T1_rl/franka_manual_safety_v002/` 已写入 runtime/fsm/panel/configs + Item1–3 md



---

## 7. 同步记录

- 本地备份：`/workspace/plan_d_work/out/*.bak_item3_20261023_190541`
- 远端备份：`.../franka_manual_safety_v002/.bak_item3_20261023_190834`
- 已推送：`plan_d_runtime.py`, `step_fsm.py`, `keyboard_panel.py`, `safety_config.json`, `geometry_calib.json`, `CLEARANCE_MERGE_ITEM3.md`, `MEASUREMENT_SPEC_ITEM2.md`, `THRESHOLD_AUDIT_ITEM1.md`
- 盒子直连 `100.118.30.122` 超时；经本机 Wangjy paramiko 成功。
