# 阈值整理 · 第 1 项：卡扣开角审计（15° vs 40°）

日期：2026-10-23（Asia/Shanghai）  
范围：仅审计，**未改任何阈值文件**  
训练场景：`isaac-b:/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/`  
Live 拉取：已 SSH 确认 `socket_print_v007_installed.usda` 铰链限位与 v002 绑定链（密钥 `id_ed25519_isaac` 拒识，改用既有诊断脚本口令通道；`id_ed25519_isaac_ts` 在 WSL 缺失）。

---

## 结论（先看这个）

| 问题 | 答案 |
|------|------|
| 「正常开角」是 15° 还是 40°？ | **两套不同含义叠在一起**：设计/评审里的「够开」约定是 **~15°（open15）**；训练场景 MatchedSocket **物理关节硬限位是 ±40°**。 |
| v002 绑的是哪套？ | MatchedSocket → `socket_franka_install_v008/socket_print_v007_installed.usda` → **Left −40…0°，Right 0…40°**。 |
| `safety_config` 里的 **16°** 是什么？ | **演示/诊断停线**（README：约 17° 几何已相交），**不是**厂家开角，也不是当前关节硬限。 |
| 面板实际用哪个？ | Plan D `keyboard_panel` **启动时用铰链 `max(|low|,|high|)+0.5` 覆盖 JSON 的 16** → 对 print_v007 实际变成 **40.5°**。 |
| 本轮建议 | **先别改阈值数字**。语义拆开：关节硬限=40°；任务「开到位」成功线≈15°（待第 4 项基线测）；16° 保留为历史干涉诊断注释，或第 2–4 项后再决定是否恢复为 soft stop。 |

---

## 1. 所有出现开角/限位的地方

### 1.1 训练场景绑定链（权威路径）

```
franka_manual_safety_v002/manual_franka_scene.usda
  └─ subLayer → ../t1_dimm_insert_train.usda
       └─ /World/MatchedSocket
            └─ references → socket_franka_install_v008/socket_print_v007_installed.usda</Asset>
                 customData.matched2308107Revision = "print_geometry_v007_installed_preview"
```

`safety_config.json` paths.socket = `/World/MatchedSocket/Asset`  
→ 面板读的铰链 = `.../Asset/Joints/{Left,Right}Hinge`。

### 1.2 角度出现清单（path · key · value · unit）

| 来源 | 键/属性 | 值 | 单位 | 角色 |
|------|---------|-----|------|------|
| **Live** `.../socket_franka_install_v008/socket_print_v007_installed.usda` `LeftHinge` | `physics:lowerLimit` / `upperLimit` | **−40 / 0** | deg（USD Revolute） | **训练场景物理硬限** |
| 同上 `RightHinge` | `physics:lowerLimit` / `upperLimit` | **0 / 40** | deg | **训练场景物理硬限** |
| 同上 hinges | `drive:angular` damping/maxForce/stiffness | 0.001 / 0.02 / 0 | — | 弱粘滞 drive（源资产）；Plan D session 再覆盖 |
| `t1_dimm_insert_train.usda` `/World/T1Safety/left_latch` | `physics:lowerLimit` / `upperLimit` | 0 / **25** | deg | **遗留 invisible cube 关节**，非 MatchedSocket |
| 同上 `right_latch` | 同上 | 0 / **25** | deg | 同上，勿当训练卡扣 |
| `.../physics_proxy_pending.usda` `LeftHinge` | lower/upper | **−15 / 0** | deg | **旧 proxy 资产**，v002 MatchedSocket **未引用** |
| 同上 `RightHinge` | lower/upper | **0 / 15** | deg | 同上；targetPosition ±15 |
| `franka_manual_safety_v002/safety_config.json` | `rules[].left_angle_abs_deg.limit` | **16** | deg | 演示红停线（文件静态值） |
| 同上 | `rules[].right_angle_abs_deg.limit` | **16** | deg | 同上 |
| 同上 | `status` | `PLAN_D_LOCAL_CALIB_DEMO_THRESHOLDS_NOT_MATERIAL_LIMITS` | — | 标明非材料极限 |
| `geometry_calib.json` | `latch_lock.closed_angle_abs_deg_max` | **3.0** | deg | **闭锁**判据（近 0），不是开角 |
| `geometry_calib.json` | （无 open_angle 字段） | — | — | 开到位成功线**尚未标定进 calib** |
| `keyboard_panel.py`（v002 / plan_d out） | 启动覆盖 `_rule["limit"]=max(abs(low),abs(high))+.5` | 对 print_v007 → **40.5** | deg | **运行时权威 UI/安全开角限** |
| 同文件注释 | — | “Opening permitted throughout composed hinge range；0.5°=solver tol” | — | 明确允许开满关节程 |
| `v002/README.md` | 文案 | **16°=干涉前诊断停线（约17°相交）**；非厂家开角/机械验收 | — | 文档权威解释 16 |
| `matched_work/.../render_matched_review.py` | state `open15` | 旋转 **15°** | deg | 设计/渲染「打开态」约定 |
| `functional_latch_revision_v1/...` | `open15_end_clear...` 等 hookmetrics | 15 相关净空 | — | 功能评审用 open15，非物理 stop |
| `ownership_confirmation` PDF 注记 | latch open envelope | — | — | **整体打开包络，不是真实铰链止挡角** |
| Plan D `plan_d_runtime.py` | 不设开角阈 | — | — | 只改 damping/maxForce/颜色 |
| `step_fsm.py` | `dual_latched` 用 `closed_angle_abs_deg_max` | 默认 3 | deg | 闭锁门禁，与开角无关 |

**未改动本轮的本地副本**（与 live 开角规则一致）：  
`/workspace/plan_d_work/out/{safety_config,geometry_calib,keyboard_panel,plan_d_runtime,step_fsm}.*`

---

## 2. 三层含义对照

| 层 | 数值 | 含义 |
|----|------|------|
| **Mesh/Joint 物理限位**（v002 MatchedSocket） | L: −40…0，R: 0…40 | PhysX 关节硬止挡；再开不动 |
| **设计「正常打开」约定** | ~**15°**（open15 / proxy ±15） | 够插拔净空的评审姿态，**不是** print_v007 的 joint limit |
| **演示安全/UI** | JSON **16°**；运行时被面板改成 **40.5°** | 16=旧干涉诊断；Plan D 后实际按关节程放行 |
| **闭锁** | \|angle\| ≤ **3°** + seated | 与开角成功线分离 |

冲突点（理解用，本轮不改文件）：  
README 要 16° 防 ~17° 干涉，但 Plan D 面板用 40.5° 覆盖 → **干涉诊断停线在现面板上基本失效**，关节可开到 40°。

---

## 3. 训练场景 v002 谁说了算

| 用途 | 权威文件 |
|------|----------|
| 场景图/资产绑定 | `manual_franka_scene.usda` → `t1_dimm_insert_train.usda` → **`socket_print_v007_installed.usda`** |
| 关节硬限 | 上述 print_v007 的 `LeftHinge`/`RightHinge` |
| 静态阈值表 | `franka_manual_safety_v002/safety_config.json` |
| 局部几何/闭锁角 | `geometry_calib.json` |
| **运行时开角红停（当前）** | `keyboard_panel.py` 对 hinge limit 的 **覆盖**（优先于 JSON 16） |
| 文档对 16° 的定义 | `README.md`（干涉诊断，非厂家开角） |

**不要**把下列当成 v002 MatchedSocket 开角：  
- `socket_*_physics_proxy_pending.usda` 的 ±15  
- `/World/T1Safety/*_latch` 的 0–25  

---

## 4. 建议（仍不改文件）

1. **关节硬限**：承认并记录为 **40°（单侧幅值）** — 与 live print_v007 一致。  
2. **不要**把「正常开」改成 40°：40 是止挡，不是作业开角。  
3. **不要**在未做第 4 项基线前，把 JSON 16 改成 15 或 40。  
4. 语义建议（供第 2–4 项落地）：  
   - `open_enough_deg` ≈ **15**（任务成功/课程 A 过关，来自 open15 + 基线）  
   - `joint_hard_stop_deg` = **40**（来自 USD）  
   - `interference_warn_deg` ≈ **16**（历史诊断；若 mesh 仍 17° 相交，应恢复为黄/红 soft，**不要**被 hinge+0.5 盖掉）  
5. 面板覆盖逻辑与 README 16° **不一致** — 列为第 2/4 项前必须决策的点：保留「开满程」还是「干涉前停」。

---

## 5. 第 2–4 项预留缺口（只记事实，不改阈）

### 第 2 项 · 测量定义缺口

- 接触力：README 写明 = **法向冲量/dt**，**不含摩擦**；不能当腕部六维力。  
- `left_press_N` / `right_press_N`：需钉死接触对（指尖 vs Latch Geometry？）、单指/双指、滤波窗、坐标系。  
- 开角：`left/right_angle_abs_deg` = **关节角绝对值**；左右符号相反（L 负开、R 正开）。  
- 铰链力矩规则 `limit: null` — 只显示不红停。  
- 采样：物理步 vs UI 刷新；红停是否「硬超限立即 / 黄持续 Ty」。

### 第 3 项 · 几何门禁可能打架

当前并存（均为演示值）：

| 量 | 值 | 出处 |
|----|-----|------|
| `tilt_deg` | 2° | safety_config + calib `tilt_tol_deg` |
| `end_difference_mm` | 1 mm | safety_config |
| `align_xy_m` | 2.5 mm | safety_config + calib `xy_tol_m` |
| `seated_depth_tol_m` | 1.5 mm | geometry_calib only |
| 门禁 | 未齐禁下压；未双锁禁松夹 | plan_d.gates |

风险：tilt / 两端高差 / XY 都在描述「插入净空」，可能重复红停或一个绿一个红。第 3 项应合成**一条插入净空判据**（或主判据+附属显示）。

### 第 4 项 · 基线应测什么（协议草稿）

在**不改阈**前提下，键盘正常流程各记峰值/稳态：

1. **拨扣**：开到可插净空时的 \|angle\|（是否≈15）、接触力、有无 17° 附近视觉/穿透。  
2. **抓取**：许可区指力、滑移、禁区是否误触。  
3. **插入**：轴向/横向接触、tilt、端高差、XY、入座深。  
4. **闭锁/松夹**：闭角、seated、松后回弹。  

用分布定黄=预警、红=停止；并决定 16 干涉线是否恢复。

---

## 6. 访问备注

- Live isaac-b：`100.118.30.122` 可达；`~/.ssh/id_ed25519_isaac` **公钥已不被接受**；bak 配置写的 `id_ed25519_isaac_ts` **文件缺失**。  
- 本审计用既有诊断口令通道只读确认 USDA/JSON；**未动 Franka、未改场景**。  
- 本地缓存：`/workspace/plan_d_work/out/`、`/workspace/dimm_diag/`（含历史 dump，与 live 开角结论一致）。

---

**Item 1 完成。** 阈值文件保持不动。等你点头再开 Item 2（测量定义）或先定「开满 40 vs 干涉停 16」产品决策。
