# 测量定义规范 · 第 2 项（MEASUREMENT_SPEC）

日期：2026-10-23（Asia/Shanghai）  
范围：**只统一定义**，不改任何阈值数字  
训练场景：`franka_manual_safety_v002`  
源码依据（优先 live）：

- `.../dimm_safety_feedback_v001/live_safety_feedback.py`（基类 `Feedback.measure`）
- `.../franka_manual_safety_v002/launch_panel.py`（覆盖 `measure` / 分项聚合）
- `.../safety_core.py`（`dimm_contact_kind` / `unexpected_robot_contact` / `SafetyEngine`）
- `.../plan_d_runtime.py`、`step_fsm.py`、`keyboard_panel.py`
- `safety_config.json`、`geometry_calib.json`

---

## 0. 根路径与刚体（固定约定）

| 角色 | Prim 路径 |
|------|-----------|
| Robot | `/World/Robot` |
| Socket 资产根（碰撞/铰链） | `/World/MatchedSocket/Asset`（代码常量 `SOCKET`） |
| Socket 父 Xform（Plan D 对齐） | `/World/MatchedSocket` |
| DIMM | `/World/MatchedDIMM`（常量 `DIMM`） |
| Left / Right 卡扣体 | `{SOCKET}/LeftLatch`、`{SOCKET}/RightLatch` |
| 卡扣碰撞几何（常用子） | `{SOCKET}/{Left\|Right}Latch/Geometry` |
| 铰链 | `{SOCKET}/Joints/{Left\|Right}Hinge` |
| 左/右指（路径子串） | `panda_leftfinger` / `panda_rightfinger`（在 Robot 下） |
| 安装基座豁免 | `.../panda_link0`（不计入 unexpected） |

PhysX：对 Socket/DIMM/Robot 等刚体挂 `PhysxContactReportAPI`（threshold=0）。  
v001 基类：`subscribe_full_contact_report_events`；v002 `launch_panel`：`subscribe_contact_report_events`（摩擦回调传空列表）。

**日志字段**：`friction_impulses_included: false` — **力一律不含摩擦冲量**。

---

## 1. 接触对象（prim 对 → 度量）

单帧接触事件字段：`a`,`b`（collider SDF 路径）、`point`、`normal`、`impulse`、`separation_mm`。

### 1.1 卡扣按压 `left_press_N` / `right_press_N`

- **计入条件**（`launch_panel.measure`）：  
  `a` 或 `b` 以 `{SOCKET}/{side}Latch` 为前缀，**且并非**「双方都在 `SOCKET` 下」（排除卡扣–基座等 socket 内轴承接触）。
- **累加量**：各合格接触点的 `normal_force_N` **求和**（标量，≥0）。
- **对手不限**：可为手指、DIMM、工具等外部体；只要触到该侧 Latch。
- **不做**：不要求必须是指尖；不区分接触面片 ID。
- **过力变色**：`plan_d_runtime.set_latch_overforce_color` 用同一 `left/right_press_N` 与 `plan_d.latch_overforce_N` 比较。

### 1.2 指力 `finger_left_N` / `finger_right_N`

- **分类**（`safety_core.dimm_contact_kind`）：一方在 `DIMM` 下，另一方路径含 `panda_leftfinger` / `panda_rightfinger`。
- **累加**：该侧所有此类接触的 `normal_force_N` **求和**。
- **不含**：指–Socket、指–桌、双指互碰（互碰走 unexpected 逻辑，见下）。

### 1.3 DIMM–环境轴向/横向 `axial_contact_N` / `lateral_contact_N`

- **仅** `dimm_contact_kind == "environment"`：一方在 DIMM，另一方**不在** Robot。  
  → **排除** 指夹接触；典型为 DIMM–Socket / DIMM–桌子等。
- 同时累加 `dimm_contact_N`（环境法向力总和，作 `contact_gated` 门控）。
- **轴**：插入轴世界方向 `axis`（见 §2）。
- `axial_contact_N += |F · axis|`（每点，再求和）  
- `lateral_contact_N += ‖F − (F·axis)axis‖`（每点，再求和；对立侧不互相抵消）

> 注意：v001 基类曾把「凡含 DIMM 的接触」都进 dimm/lat（含手指）；**v002 launch 已按 kind 拆开，以 launch 为准。**

### 1.4 意外机器人接触 `unexpected_robot_contact_N`

- **判定** `unexpected_robot_contact(a,b, robot, dimm, socket, step)`：  
  - 双方都在 Robot → True（自碰）  
  - 仅 `panda_link0` 触外部 → False  
  - 指 + DIMM 且 step∈[1,6] → False（合法抓取）  
  - 指 + `{socket}/{Left\|Right}Latch` 且 step==0 → False（合法拨扣）  
  - 其余「Robot 移动件 ↔ 外部」→ True  
- **累加**：上述为 True 的接触的 `normal_force_N` 求和。
- **键盘 Plan D**：`keyboard_panel` 把该函数包成  
  `old(…,step=0) and old(…,step=1)` → **更宽**：只有两套步骤解释都算意外才计入（减少 free jog 误报）。

### 1.5 其它接触相关

| 键 | 定义 |
|----|------|
| `penetration_mm` | `max(socket 内重叠 mm, max(0, −min_separation_mm))`；socket–socket 只更新重叠，不进力 |
| `forbidden_contact` | 指–DIMM 接触点变到 **DIMM 局部**；落在禁区盒 **或** 不在任何许可盒 → 1.0，否则 0.0（需 `grasp_regions.calibrated`） |
| `dimm_moment_Nm` | 环境接触（kind=environment）对 DIMM 原点力矩向量模长 |
| `latch_overforce` | bool/0–1：任一侧 press ≥ `latch_overforce_N` |

**金手指 `Gold_*`**：Plan D 关碰撞 → **不出现在接触报告**，不参与力/成功。

---

## 2. 坐标系与轴

### 2.1 帧

| 帧 | 用途 |
|----|------|
| **World** | 接触点、冲量/力向量、机器人 EE、门禁「下压」= world −Z |
| **Socket Asset / Socket 父** | 插入轴、对齐；见下「不一致」 |
| **DIMM local** | 抓取许可/禁区盒；`long_axis=[1,0,0]`，`normal_axis=[0,1,0]` |
| **Socket local（calib）** | `insertion_axis=[0,0,1]`，`entry_point`，`seated_reference` |

### 2.2 插入轴（axial）

- 基类/launch：`axis = R_socket_asset · [0,0,1]`（世界系单位向量）  
  即 **Socket Asset 局部 +Z**。
- `geometry_calib.socket_local.insertion_axis` / `safety_config.geometry.socket_insertion_axis` 同为 `[0,0,1]`。
- **横向** = 力在垂直于 `axis` 的平面内分量。

### 2.3 对齐 `align_xy_m`（Plan D）

- 用 **`/World/MatchedSocket`（父）** 的 world→local，把 DIMM 原点变到 socket 局部。  
- `align_xy_m = √((x−entry_x)²+(y−entry_y)²)`（忽略 Z）  
- `seated_depth_m = z_local − seated_reference_z`  
- `seated_ok`：`|depth|≤seated_depth_tol` 且 `dxy ≤ 2×xy_tol`（代码如此，偏松）

**已知不一致（Item3 相关）**：力/tilt 基类用 **Asset** 姿态；Plan D 对齐用 **MatchedSocket 父**。若父子有相对变换，轴与 XY 会漂。规范建议：统一到 `socket_asset_prim` 或显式写清 offset。

### 2.4 倾斜 `tilt_deg`（当前主路径 = 基类）

对 DIMM 姿态：

1. `board_normal = R_dimm · [0,1,0]`  
2. `long_axis = R_dimm · [1,0,0]`  
3. `t1 = acos( |board_normal · (R_socket·[0,1,0])| )`（度）— 板面法向相对 socket Y  
4. `t2 = asin( |long_axis · axis| )`（度）— 长轴相对插入轴的「立起」偏离  
5. **`tilt_deg = max(t1, t2)`**

`contact_gated: true`：仅当 `dimm_contact_N > contact_gate_N`（0.05 N）时规则才激活。

### 2.5 Plan D 备用倾斜（仅主 `tilt_deg is None` 时）

`tilt_board_to_socket_z_deg = |90° − acos(|n_board · z_socket|)|`  
**与基类公式不同**；键盘在缺省时写入 `tilt_deg`。规范：**优先基类公式；备用仅作 fallback，Item3 应禁止 silently 混用。**

### 2.6 两端高度差 `end_difference_mm`

- `delta = | long_axis · axis | × dimm_length × 1000`  
- `dimm_length = 0.13335 m`（半长×2 的全长投影）  
- 几何意义：DIMM 两端沿插入轴的高度差（mm）；**不是**左右 hook 入座深度差的独立测距。

### 2.7 卡扣开角

- 刚体视图位姿 + 铰链 `localRot0/1`、`axis` → 相对转角（度，**有符号**）  
- `left_angle_deg` / `right_angle_deg` 存 `self.last`  
- 上报：`left/right_angle_abs_deg = abs(signed)`  
- Live 关节：Left 开向负、Right 开向正（Item1：L −40…0，R 0…40）  
- **闭锁**：`|angle_abs| ≤ closed_angle_abs_deg_max`（calib 3°）且 `seated_ok`，**角≈0 ≠ 已锁**

---

## 3. 力分量约定

| 量 | 公式 / 约定 |
|----|-------------|
| 冲量→力 | `force = impulse / dt`（向量）；标量冲量时 `force = normal * impulse/dt` |
| v002 时间对齐 | `impulse *= physics_dt / reported_dt` 后再 `/dt` |
| `normal_force_N` | `|force · normal|`（≥0） |
| 摩擦 | **不含**；UI/日志明确声明 |
| 符号 | press/finger/unexpected/axial 用的法向贡献为 **非负累加**；力矩用对 body 的 signed `F` |
| 非腕力计 | 法向接触代理，≠ Franka 腕部六维力/驱动关节力矩 |

铰链力矩 `left/right_torque_Nm`：对铰链轴 `u` 的 `Σ (r×F_signed)·u`；**limit=null**，只显示。

---

## 4. 单指 / 双指与夹力门禁

| 项 | 行为 |
|----|------|
| 左右指力 | **分通道**求和，互不合并 |
| 接触减速 | `max(finger_left_N, finger_right_N) > 0.05` → 降速 jog |
| 滑移参考 | **双侧** `>0.05` 时锁 `slip_reference`（手系下 DIMM 相对位移） |
| `grip_force_limit_N` | **`max(fl, fr) ≥ limit` → `block_close`**（禁止继续合爪）；**不是 sum** |
| 缺省 | 一侧缺失当 0 |

---

## 5. 采样、峰值、回差、exposure

| 项 | 定义 |
|----|------|
| 采样触发 | 每个 PhysX `physics_step`（`dt` 来自回调） |
| 候选场景 | README：诊断向 ~1000 Hz；以实际 timeline/PhysX 为准 |
| 瞬时值 | 当步 contacts 聚合后的 metrics |
| 峰值 `peak` | `SafetyEngine` / `TrafficLight`：`max(|value|)` 在规则激活期间 |
| 红 | `\|v\|/limit ≥ 1` **立即**；锁存，需人工 ack |
| 黄 | `≥ warning_ratio(0.8)`；清到 `< clear_ratio(0.7)`；Plan D 另需黄/绿最短 **0.05 s** |
| `exposure_s` | 该规则持续处于 YELLOW 或 RED 的累计时间（秒） |
| 灰 | 本步不在 `steps`、limit=null、值无效、或 contact_gated 未满足 |
| 门禁 vs 灯 | StepFSM：红锁存 / overforce → halt，禁下压与松夹；对齐/双锁/夹力另有方向门禁 |

---

## 6. 指标一览（定义层，阈值数字仅作现状引用）

| key | 单位 | 接触对/几何 | 聚合 | 有符号？ | limit 现状 |
|-----|------|-------------|------|----------|------------|
| left/right_press_N | N | 外部↔该侧 Latch | Σ n | 否 | 3（演示） |
| left/right_angle_abs_deg | deg | 铰链 | abs(signed) | 上报非负 | 16 JSON；面板可盖成 40.5 |
| left/right_torque_Nm | N·m | 同上 + 杠杆 | Σ | 有 | **null** |
| finger_left/right_N | N | 指↔DIMM | Σ n | 否 | **null** |
| axial_contact_N | N | DIMM↔非Robot | Σ \|F·axis\| | 否 | 5 |
| lateral_contact_N | N | 同上 | Σ ‖F⊥axis‖ | 否 | 2 |
| tilt_deg | deg | 姿态 | max(t1,t2) | 否 | 2（需接触门控） |
| end_difference_mm | mm | 姿态 | 投影高差 | 否 | 1（需接触门控） |
| align_xy_m | m | 位姿 | XY 距 entry | 否 | 0.0025 |
| dimm_moment_Nm | N·m | 环境接触 | ‖Σ r×F‖ | 否 | **null** |
| slip_mm | mm | 手–DIMM 相对 | 相对参考位移 | 否 | **null** |
| forbidden_contact | bool | 指点∈禁区 | 0/1 | — | 0.5 |
| penetration_mm | mm | 重叠诊断 | max | 否 | 0.1 |
| tracking_error_mm | mm | hand FK vs goal | ‖Δp‖ | 否 | 3（面板可抬到 1000） |
| ee_speed_mm_s | mm/s | hand | ‖Δp‖/dt | 否 | **null** |
| unexpected_robot_contact_N | N | 见 §1.4 | Σ n | 否 | 0.05 |
| latch_overforce | bool | press vs 阈 | or | — | 0.5 |

---

## 7. 缺口 → 留给第 4 项基线（及第 3 项几何合并）

1. **null 限且训练/安全会用到**：`finger_*`、`torque_*`、`slip_mm`、`dimm_moment_Nm`、`ee_speed` — 需正常流程峰值分布再定黄/红。  
2. **开角语义三分裂**（Item1）：关节硬限 40° / 作业够开 ~15° / 干涉诊断 16° — 测量已有 `angle_abs`，缺 `open_enough` 成功线字段。  
3. **tilt 双公式**：基类 vs Plan D fallback — 基线日志必须标明用的是哪一个。  
4. **Socket 父 vs Asset 帧** 对齐与插入轴是否一致 — 应用同一 prim 重测 `align_xy` / tilt。  
5. **`end_difference_mm` 是否等于真实两端入座差** — 应用两端接触点 Z 差对照。  
6. **latch press 对手未钉死**（任意外部）— 基线应分「仅指尖」「含 DIMM」子集统计。  
7. **grip 用 max 非 sum** — 基线报告两侧分位，避免单指顶满误伤。  
8. **contact_gate / 减速阈 0.05 N** 与噪声 — 空载噪声地板。  
9. **unexpected 键盘 AND 包装** — 训练 env 是否沿用同一宽松定义，需产品拍板。

---

## 8. 草稿：`safety_config.measurement` schema（**勿直接落盘**）

```json
{
  "measurement": {
    "schema_version": 1,
    "contact_api": "physx_contact_report",
    "include_friction_impulse": false,
    "force_from": "impulse_over_dt",
    "normal_force_def": "abs_dot_force_normal",
    "roots": {
      "robot": "/World/Robot",
      "socket_asset": "/World/MatchedSocket/Asset",
      "socket_frame_for_align": "/World/MatchedSocket",
      "dimm": "/World/MatchedDIMM"
    },
    "axes": {
      "insertion_in_socket_local": [0, 0, 1],
      "dimm_long_local": [1, 0, 0],
      "dimm_normal_local": [0, 1, 0],
      "world_down_for_gate": [0, 0, -1]
    },
    "channels": {
      "left_press_N": {
        "pairs": "external_vs_prefix:/World/MatchedSocket/Asset/LeftLatch",
        "exclude": "both_under_socket_asset",
        "reduce": "sum_normal_force"
      },
      "right_press_N": {
        "pairs": "external_vs_prefix:/World/MatchedSocket/Asset/RightLatch",
        "exclude": "both_under_socket_asset",
        "reduce": "sum_normal_force"
      },
      "finger_left_N": {
        "pairs": "dimm_vs_path_substr:panda_leftfinger",
        "reduce": "sum_normal_force"
      },
      "finger_right_N": {
        "pairs": "dimm_vs_path_substr:panda_rightfinger",
        "reduce": "sum_normal_force"
      },
      "axial_contact_N": {
        "pairs": "dimm_vs_non_robot",
        "reduce": "sum_abs_dot_force_insertion_axis"
      },
      "lateral_contact_N": {
        "pairs": "dimm_vs_non_robot",
        "reduce": "sum_norm_force_perp_insertion_axis"
      },
      "unexpected_robot_contact_N": {
        "predicate": "safety_core.unexpected_robot_contact",
        "keyboard_wrapper": "and_step0_step1",
        "reduce": "sum_normal_force"
      }
    },
    "pose_metrics": {
      "tilt_deg": {
        "primary": "max(acos(|n_board·Y_sock|), asin(|long·insert|))",
        "fallback_plan_d": "abs(90-acos(|n_board·Z_sock|))",
        "contact_gated_by": "dimm_contact_N"
      },
      "end_difference_mm": "abs(dot(long_axis, insert_axis))*dimm_length_m*1000",
      "align_xy_m": "hypot(xy(dimm_origin in socket_frame) - entry_xy)",
      "angle_abs_deg": "abs(hinge_signed_deg)"
    },
    "grip_limit_reduce": "max",
    "sampling": {
      "on": "physics_step",
      "traffic_light": {
        "warning_ratio": 0.8,
        "warning_clear_ratio": 0.7,
        "yellow_min_duration_s": 0.05,
        "green_min_duration_s": 0.05,
        "red_immediate": true,
        "red_requires_ack": true
      },
      "exposure_s": "time_in_yellow_or_red"
    }
  }
}
```

---

**Item 2 完成。** 未改阈值、未开 Item 3。  
文件：`/workspace/plan_d_work/out/MEASUREMENT_SPEC_ITEM2.md`
