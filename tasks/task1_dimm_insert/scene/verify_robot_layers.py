#!/usr/bin/env python3
import re
from pathlib import Path
MANUAL = Path("/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/manual_franka_scene.usda")
TRAIN = Path("/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/t1_dimm_insert_train.usda")
RISER = Path("/mnt/data2/wjy_SERL/franka_sim/outputs/T1/riser_io_preview.usda")
ROBOT = Path("/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_latch_open_test_v1/robot_cache/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd")

def link_xform_hits(path):
    rows = path.read_text(encoding="utf-8").splitlines()
    hits = []
    in_r = False
    d = 0
    seen = False
    for i, l in enumerate(rows, 1):
        if (not in_r) and ('def Xform "Robot"' in l or 'over "Robot"' in l):
            in_r = True
            d = 0
            seen = False
        if not in_r:
            continue
        if "{" in l or "}" in l:
            d += l.count("{") - l.count("}")
            seen = True
        if re.search(r"xformOp:(translate|orient|transform|scale|rotate)", l):
            recent = "\n".join(rows[max(0, i - 30):i])
            if 'over "panda_' in recent:
                hits.append((i, l.strip()))
        if seen and d == 0:
            in_r = False
    return hits

ok = True
print("robot_usd_exists", ROBOT.exists())
print("props_link0", (ROBOT.parent / "Props/panda_link0.usd").exists())
print("materials", (ROBOT.parent / "Materials/Materials.usd").exists())
for p, exp in [(MANUAL, True), (TRAIN, False), (RISER, False)]:
    defs = len(re.findall(r'def Xform "Robot"|over "Robot"', p.read_text(encoding="utf-8")))
    hits = link_xform_hits(p)
    print("%s: robot_defs=%s link_xform_hits=%s" % (p, defs, len(hits)))
    for h in hits:
        print(" ", h)
    if exp and defs < 1:
        ok = False
    if (not exp) and defs != 0:
        ok = False
    if hits:
        ok = False
raise SystemExit(0 if ok else 1)
