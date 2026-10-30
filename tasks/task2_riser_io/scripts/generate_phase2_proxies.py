#!/usr/bin/env python3
"""Generate Phase2 proxy USDA (S2 nut plate, S3 M4 screws, S4 screw tray) and patch mother."""
from __future__ import annotations
from pathlib import Path
import json
import shutil
from datetime import datetime

ROOT = Path("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io")
SCENE = ROOT / "scene"
ASSETS = SCENE / "assets" / "task2_wire" / "usd" / "phase2"
MOTHER = SCENE / "riser_io_preview.usda"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

HOLES = [
    ("Hole_1", -0.06, -0.03),
    ("Hole_2",  0.06, -0.03),
    ("Hole_3", -0.06,  0.03),
    ("Hole_4",  0.06,  0.03),
]
PCB4_T = (-0.050, -0.050, 0.750)
TABLE_Z = 0.750

M4_SHANK_R = 0.0020
M4_CLEAR_R = 0.00225
M4_TAP_R = 0.00165
M4_HEAD_R = 0.0035
M4_HEAD_H = 0.0030
M4_SHANK_H = 0.0120

PLATE_T = 0.006
PLATE_X = 0.140
PLATE_Y = 0.080

TRAY_T = (0.22, -0.28, TABLE_Z)
TRAY_OUTER = (0.120, 0.080, 0.012)
NEST_SPACING_X = 0.028
NEST_SPACING_Y = 0.028
NEST_R = 0.006
NEST_DEPTH = 0.006
NEST_XY = [
    (-NEST_SPACING_X / 2, -NEST_SPACING_Y / 2),
    ( NEST_SPACING_X / 2, -NEST_SPACING_Y / 2),
    (-NEST_SPACING_X / 2,  NEST_SPACING_Y / 2),
    ( NEST_SPACING_X / 2,  NEST_SPACING_Y / 2),
]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} B)")


def nut_plate_usda() -> str:
    hole_prims = []
    for name, x, y in HOLES:
        hole_prims.append(f"""
    def Cylinder "{name}_TapProxy" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        customData = {{
            string role = "M4_threaded_hole_proxy"
            string note = "tap-drill radius proxy; not true helical thread"
        }}
        color3f[] primvars:displayColor = [(0.15, 0.15, 0.18)]
        double height = {PLATE_T + 0.0005}
        double radius = {M4_TAP_R}
        bool physics:collisionEnabled = 0
        quatf xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:translate = ({x}, {y}, {PLATE_T / 2})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}

    def Cylinder "{name}_ClearGuide" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        customData = {{
            string role = "clearance_guide_visual"
        }}
        color3f[] primvars:displayColor = [(0.35, 0.55, 0.75)]
        double height = 0.001
        double radius = {M4_CLEAR_R}
        bool physics:collisionEnabled = 0
        double3 xformOp:translate = ({x}, {y}, {PLATE_T + 0.0006})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
""")
    return f"""#usda 1.0
(
    defaultPrim = "NutPlate_M4"
    metersPerUnit = 1
    upAxis = "Z"
    customLayerData = {{
        string purpose = "Phase2 S2 threaded nut-plate proxy under PCB"
        string metric = "M4"
        string hole_layout = "from pcb_4_terminal MountHole_1..4 (+-0.06, +-0.03)"
        string generated = "{TS}"
    }}
)

def Xform "NutPlate_M4" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
)
{{
    bool physics:kinematicEnabled = 1
    bool physics:rigidBodyEnabled = 1
    customData = {{
        string id = "S2"
        string metric = "M4"
    }}

    def Cube "PlateBody" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        color3f[] primvars:displayColor = [(0.55, 0.55, 0.58)]
        double size = 1
        bool physics:collisionEnabled = 1
        float3 xformOp:scale = ({PLATE_X}, {PLATE_Y}, {PLATE_T})
        double3 xformOp:translate = (0, 0, {PLATE_T / 2})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
{''.join(hole_prims)}
}}
"""


def m4_screw_usda() -> str:
    return f"""#usda 1.0
(
    defaultPrim = "Screw_M4"
    metersPerUnit = 1
    upAxis = "Z"
    customLayerData = {{
        string purpose = "Phase2 S3 graspable M4 screw proxy"
        string metric = "M4"
        float shank_length_m = {M4_SHANK_H}
        float head_height_m = {M4_HEAD_H}
        string generated = "{TS}"
    }}
)

def Xform "Screw_M4" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    bool physics:rigidBodyEnabled = 1
    bool physics:kinematicEnabled = 0
    float physics:mass = 0.004
    vector3f physics:angularVelocity = (0, 0, 0)
    vector3f physics:velocity = (0, 0, 0)
    customData = {{
        string id = "S3"
        string metric = "M4"
        string grasp_hint = "prefer head cylinder"
    }}

    def Cylinder "Shank" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        color3f[] primvars:displayColor = [(0.72, 0.72, 0.75)]
        double height = {M4_SHANK_H}
        double radius = {M4_SHANK_R}
        bool physics:collisionEnabled = 1
        double3 xformOp:translate = (0, 0, {M4_SHANK_H / 2})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}

    def Cylinder "Head" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        color3f[] primvars:displayColor = [(0.85, 0.75, 0.25)]
        double height = {M4_HEAD_H}
        double radius = {M4_HEAD_R}
        bool physics:collisionEnabled = 1
        double3 xformOp:translate = (0, 0, {M4_SHANK_H + M4_HEAD_H / 2})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
}}
"""


def screw_tray_usda() -> str:
    nests = []
    for i, (x, y) in enumerate(NEST_XY, 1):
        nests.append(f"""
    def Xform "Nest_{i}"
    {{
        double3 xformOp:translate = ({x}, {y}, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        customData = {{
            int screw_index = {i}
        }}

        def Cylinder "PocketVisual"
        {{
            color3f[] primvars:displayColor = [(0.25, 0.25, 0.28)]
            double height = {NEST_DEPTH}
            double radius = {NEST_R}
            double3 xformOp:translate = (0, 0, {TRAY_OUTER[2] - NEST_DEPTH / 2})
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}

        def Cylinder "PocketCollider" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            color3f[] primvars:displayColor = [(0.2, 0.2, 0.22)]
            double height = 0.001
            double radius = {NEST_R}
            bool physics:collisionEnabled = 0
            double3 xformOp:translate = (0, 0, {TRAY_OUTER[2] - NEST_DEPTH})
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}
""")
    return f"""#usda 1.0
(
    defaultPrim = "ScrewTray"
    metersPerUnit = 1
    upAxis = "Z"
    customLayerData = {{
        string purpose = "Phase2 S4 screw tray with 4 nest slots"
        string metric = "M4"
        string generated = "{TS}"
    }}
)

def Xform "ScrewTray" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
)
{{
    bool physics:kinematicEnabled = 1
    bool physics:rigidBodyEnabled = 1
    customData = {{
        string id = "S4"
    }}

    def Cube "TrayBody" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        color3f[] primvars:displayColor = [(0.40, 0.32, 0.22)]
        double size = 1
        bool physics:collisionEnabled = 1
        float3 xformOp:scale = ({TRAY_OUTER[0]}, {TRAY_OUTER[1]}, {TRAY_OUTER[2]})
        double3 xformOp:translate = (0, 0, {TRAY_OUTER[2] / 2})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
{''.join(nests)}
}}
"""


def patch_mother() -> None:
    bak = MOTHER.with_suffix(MOTHER.suffix + f".bak_pre_phase2proxy_{TS}")
    shutil.copy2(MOTHER, bak)
    print(f"backup {bak}")
    text = MOTHER.read_text(encoding="utf-8")

    nut_z = PCB4_T[2] - PLATE_T
    screw_z0 = TRAY_T[2] + (TRAY_OUTER[2] - NEST_DEPTH) + 0.0005

    screw_refs = []
    for i, (nx, ny) in enumerate(NEST_XY, 1):
        sx = TRAY_T[0] + nx
        sy = TRAY_T[1] + ny
        screw_refs.append(f"""
def "Screw_{i}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI"]
    prepend references = @./assets/task2_wire/usd/phase2/m4_screw.usda@
    customData = {{
        string id = "S3"
        int index = {i}
        string nest = "ScrewTray/Nest_{i}"
        string metric = "M4"
    }}
)
{{
    bool physics:rigidBodyEnabled = 1
    bool physics:kinematicEnabled = 0
    float physics:mass = 0.004
    vector3f physics:angularVelocity = (0, 0, 0)
    vector3f physics:velocity = (0, 0, 0)
    quatf xformOp:orient = (1, 0, 0, 0)
    float3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = ({sx:.6f}, {sy:.6f}, {screw_z0:.6f})
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
}}
""")

    block = f"""
# ==== Phase2 proxy geometry (S2/S3/S4) generated {TS} ====
# Hole layout ASSUMED from pcb_4_terminal MountHole_1..4: (+-0.06, +-0.03) m, r=0.002
# Metric default: M4. Phase1 wires untouched.

def "NutPlate_M4" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
    prepend references = @./assets/task2_wire/usd/phase2/nut_plate_m4.usda@
    customData = {{
        string id = "S2"
        string align = "coaxial with /PCB_4_Terminal MountHole_1..4"
        string metric = "M4"
    }}
)
{{
    bool physics:kinematicEnabled = 1
    bool physics:rigidBodyEnabled = 1
    quatf xformOp:orient = (1, 0, 0, 0)
    float3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = ({PCB4_T[0]}, {PCB4_T[1]}, {nut_z:.6f})
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
}}

def "ScrewTray" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
    prepend references = @./assets/task2_wire/usd/phase2/screw_tray.usda@
    customData = {{
        string id = "S4"
        string fills = "ResetTray role / new ScrewTray"
        string metric = "M4"
    }}
)
{{
    bool physics:kinematicEnabled = 1
    bool physics:rigidBodyEnabled = 1
    quatf xformOp:orient = (1, 0, 0, 0)
    float3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = ({TRAY_T[0]}, {TRAY_T[1]}, {TRAY_T[2]})
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
}}
{''.join(screw_refs)}
# ==== end Phase2 proxy ====
"""

    old_tray = """    def \"ResetTray\"
    {
    }"""
    new_tray = f"""    def \"ResetTray\"
    {{
        customData = {{
            string note = "Phase2 S4: screw nests live on /ScrewTray (sibling); ResetTray kept as alias marker"
            string see = "/ScrewTray"
            string generated = "{TS}"
        }}
        def Cube \"TrayAliasMarker\"
        {{
            color3f[] primvars:displayColor = [(0.3, 0.5, 0.3)]
            double size = 1
            token visibility = \"invisible\"
            float3 xformOp:scale = (0.01, 0.01, 0.01)
            double3 xformOp:translate = ({TRAY_T[0]}, {TRAY_T[1]}, {TRAY_T[2]})
            uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]
        }}
    }}"""
    if old_tray not in text:
        raise SystemExit("ResetTray empty block not found — abort patch")
    text = text.replace(old_tray, new_tray, 1)

    marker = '\ndef "Render"'
    if marker in text:
        text = text.replace(marker, block + marker, 1)
    else:
        text = text.rstrip() + "\n" + block

    MOTHER.write_text(text, encoding="utf-8")
    print(f"patched {MOTHER}")


def update_docs() -> None:
    bom = ROOT / "PRINT_BOM.md"
    bak = ROOT / f"PRINT_BOM.md.bak_pre_phase2proxy_{TS}"
    if bom.exists():
        shutil.copy2(bom, bak)
    bom.write_text(f"""# PRINT_BOM — task2_riser_io (Phase 1 + Phase 2 proxies)

Aligned with **TASK_FLOW.md** (wire then screws). Proxy sim assets generated **{TS}** (default **M4**).

## Assumed hole layout (from USDA parse, no Kit)

Source: `assets/task2_wire/usd/pcb_4_terminal.usd` `MountHole_1..4`

| Index | Local XY (m) | Clearance r (asset) |
|-------|----------------|---------------------|
| 1 | (-0.06, -0.03) | 0.002 |
| 2 | (+0.06, -0.03) | 0.002 |
| 3 | (-0.06, +0.03) | 0.002 |
| 4 | (+0.06, +0.03) | 0.002 |

Pitch: **dX = 120 mm**, **dY = 60 mm**.
World: holes inherit `/PCB_4_Terminal` translate `(-0.050, -0.050, 0.750)` e.g. Hole1 world XY `(-0.110, -0.080)`.
`/NutPlate_M4` same XY, Z = `0.750 - 0.006`.

## Phase 2 printable / proxy dims (M4)

| ID | Part | Sim path | Print dims (mm) | Notes |
|----|------|----------|-----------------|-------|
| S2 | Nut plate + 4x M4 tap proxies | `scene/assets/task2_wire/usd/phase2/nut_plate_m4.usda` | Plate **140 x 80 x 6**; holes on 120x60 pitch; tap ~Ø3.3; clear guide Ø4.5 | Kinematic; not true helix |
| S3 | Screw x4 | `.../phase2/m4_screw.usda` -> `/Screw_1..4` | Shank Ø**4** x L**12**; pan head Ø**7** x H**3**; ~4 g | Dynamic RB on tray |
| S4 | Screw tray 4 nests | `.../phase2/screw_tray.usda` -> `/ScrewTray` | Outer **120 x 80 x 12**; nest Ø**12** x depth**6**; nest pitch 28 mm | World `(0.22, -0.28, 0.75)`; `/ResetTray` alias |
| S1 | PCB clearance | Partial Task1PCB + MountHole ref | Same 120x60 / Ø4-4.5 | Verify in Isaac |
| S5 | PCB holder | existing | — | Keep kinematic |

## Phase 1 (unchanged)

Wires / rings / TE left in mother.

## Open in Isaac

`/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/riser_io_preview.usda`
""", encoding="utf-8")
    print("updated PRINT_BOM.md")

    report = {
        "generated": TS,
        "metric": "M4",
        "hole_layout_source": "pcb_4_terminal.usd MountHole_1..4",
        "holes_local_m": [{"name": n, "x": x, "y": y} for n, x, y in HOLES],
        "pitch_m": {"dx": 0.12, "dy": 0.06},
        "pcb4_world": list(PCB4_T),
        "nut_plate_world": [PCB4_T[0], PCB4_T[1], PCB4_T[2] - PLATE_T],
        "screw_tray_world": list(TRAY_T),
        "assets": {
            "S2": str(ASSETS / "nut_plate_m4.usda"),
            "S3": str(ASSETS / "m4_screw.usda"),
            "S4": str(ASSETS / "screw_tray.usda"),
        },
        "mother_prims": ["NutPlate_M4", "ScrewTray", "Screw_1", "Screw_2", "Screw_3", "Screw_4"],
        "phase1_wires": "untouched",
        "open": str(MOTHER),
    }
    write(SCENE / "PHASE2_PROXY_REPORT.json", json.dumps(report, indent=2) + "\n")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ROOT / "scripts").mkdir(parents=True, exist_ok=True)
    write(ASSETS / "nut_plate_m4.usda", nut_plate_usda())
    write(ASSETS / "m4_screw.usda", m4_screw_usda())
    write(ASSETS / "screw_tray.usda", screw_tray_usda())
    patch_mother()
    update_docs()
    text = MOTHER.read_text(encoding="utf-8")
    for token in ["NutPlate_M4", "ScrewTray", "Screw_1", "Screw_4", "ResetTray"]:
        assert token in text, token
    wire_hits = sum(1 for w in ["wire1", "wire2", "FlexibleCable", "ring_terminal", "TE_L26620011"] if w in text)
    print("phase1 token hits", wire_hits)
    print("DONE", TS)


if __name__ == "__main__":
    main()