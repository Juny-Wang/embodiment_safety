from pathlib import Path
import json,shutil,ast,difflib,hashlib,zipfile
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io');S=R/'scene';O=S/'printable_v001';B=O/'backups';B.mkdir(exist_ok=True)
def update(p,text):
 if p.exists() and not (B/p.name).exists():shutil.copy2(p,B/p.name)
 p.write_text(text)
cfg=json.loads((S/'safety_config.json').read_text())
g=json.loads((O/'geometry_config.json').read_text())
cfg['status']='PRINTABLE_CUSTOM_D10_P2_PROTOTYPE_NOT_PHYSICS_ACCEPTED'
cfg['warning_only']=True
cfg['paths'].update(board='/PCB_4_Terminal',pcb_holder='/PCB_holder',nut_plate='/NutPlate_Coarse',
 seat_targets=g['seat_paths'],ring_paths=g['ring_paths'],screws=g['screw_paths'],nut_paths=g['nut_paths'],
 selected_screw='/Screw_1',phase2_note='Custom D10 P2 true helical printable meshes; not M4, not runtime accepted')
cfg['paths'].pop('stud_a',None);cfg['paths'].pop('stud_b',None)
cfg['plan_d']['gates']['overforce_terminate']=False
cfg['geometry_config_file']='printable_v001/geometry_config.json'
cfg['step_fsm']['note']='Phase1 wires then Phase2 printable coarse screws; no automatic success from step selection'
cfg['control']['ik_position_tolerance_m']=.0001
cfg['control']['ik_rotation_tolerance_rad']=.001745329252
for rule in cfg['rules']:
 if rule['key']=='wire_joint_force':rule['label']='Wire joint reaction (unavailable until measured)'
 if rule['key']=='screw_torque_Nm':rule['label']='Thread torque (unavailable; not inferred from drive maxForce)'
update(S/'safety_config.json',json.dumps(cfg,indent=2,ensure_ascii=False)+'\n')
cal={'schema_version':2,'status':'CAD_FRAMES_MEASURED_KIT_UNVERIFIED','units':'m','metric':'CUSTOM_D10_P2_NOT_ISO',
 'alignment':{'xy_tol_m':.0003,'tilt_tol_deg':1.0,'source':'candidate geometric budget; unvalidated operation gate'},
 'seat_world_m':g['seat_world_m'],'seat_paths':g['seat_paths'],'ring_paths':g['ring_paths'],'screw_paths':g['screw_paths'],
 'ring_thickness_m':.003,'screw_head_underside_local_z_m':.018,'thread_pitch_m':.002,'physics_ready':False}
update(S/'geometry_calib.json',json.dumps(cal,indent=2)+'\n')
# Remove uncalibrated automatic pass; provide CAD-frame measurements without claiming Kit accuracy.
p=S/'plan_d_runtime.py';s=p.read_text()
a=s.index('def evaluate_insertion_clearance(');b=s.index('\ndef set_latch_overforce_color',a)
s=s[:a]+'''def evaluate_insertion_clearance(stage, calib, safety, metrics=None):
    m=metrics or {}
    xy=m.get("align_xy_m");tilt=m.get("tilt_deg")
    ok=xy is not None and tilt is not None and xy<=calib["alignment"]["xy_tol_m"] and tilt<=calib["alignment"]["tilt_tol_deg"]
    return {"insertion_clearance_ok":ok,
            "insertion_clearance_reason":"cad_alignment_only_no_collision_proof" if ok else "unaligned_or_measurement_unavailable"}

def compute_alignment_metrics(stage, calib, safety):
    import math
    from pxr import UsdGeom,Gf
    i=int(safety.get("selected_index",0))
    phase=int(safety.get("selected_phase",1))
    target=stage.GetPrimAtPath(calib["seat_paths"][i])
    source=stage.GetPrimAtPath(calib["ring_paths"][i] if phase==1 else calib["screw_paths"][i])
    if not source or not target:return {"align_xy_m":None,"tilt_deg":None,"alignment_source":"missing_prim"}
    xc=UsdGeom.XformCache()
    sm=xc.GetLocalToWorldTransform(source);tm=xc.GetLocalToWorldTransform(target)
    relative=tm.GetInverse().Transform(sm.ExtractTranslation())
    axis=sm.TransformDir(Gf.Vec3d(0,0,1)).GetNormalized()
    target_axis=tm.TransformDir(Gf.Vec3d(0,0,1)).GetNormalized()
    return {"align_xy_m":math.hypot(relative[0],relative[1]),
            "tilt_deg":math.degrees(math.acos(max(-1.,min(1.,abs(Gf.Dot(axis,target_axis)))))),
            "axial_offset_m":float(relative[2]),"alignment_source":"composed_USD_live_refresh_required"}

''' +s[b:]
s=s.replace('targets = [prim]\n        for child in prim.GetChildren():\n            targets.append(child)','from pxr import Usd\n        targets = list(Usd.PrimRange(prim))')
s=s.replace('paths.get("nut_plate", "/NutPlate_M4")','paths.get("selected_screw", "/Screw_1")')
s=s.replace('("grip_force_N", paths.get("screw_tray", "/ScrewTray"), "grip_force_N"),','')
ast.parse(s);update(p,s)
p=S/'step_fsm.py';s=p.read_text().replace('return True, "align_uncalibrated_soft_pass"','return False, "alignment_measurement_unavailable"')
update(p,s)
# Fix contact data interpretation in the existing panel, scoped to Task2.
p=S/'keyboard_panel.py';s=p.read_text()
s=s.replace('from pxr import UsdPhysics, PhysicsSchemaTools','from pxr import UsdPhysics, PhysicsSchemaTools\nimport numpy as np')
s=s.replace('        self.metrics = {}','        self.metrics = {}\n        self._physics_dt = None\n        self._contact_peaks = {}\n        self._physics_serial = 0\n        self._last_consumed_serial = -1\n        self._contact_points = []')
s=s.replace('        self._hook_contacts()','''        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and (str(prim.GetPath()).startswith("/World/Robot") or "ring_terminal" in str(prim.GetPath()) or str(prim.GetPath()).startswith("/Screw_")):
                prim.AddAppliedSchema("PhysxContactReportAPI")
                from pxr import Sdf
                prim.CreateAttribute("physxContactReport:threshold",Sdf.ValueTypeNames.Float).Set(0.)
        self._physics_subscription=omni.physx.get_physx_interface().subscribe_physics_step_events(self._on_physics_step)
        self._hook_contacts()''')
s=s.replace('                    ui.Label("Step")','''                    ui.Label("Object index (wire / ring / screw share the same index)")
                    self.object_combo=ui.ComboBox(0,"1","2","3","4")
                    ui.Label("WARNING ONLY | Print prototype | Torque and wire tension uncalibrated")
                    ui.Label("Step")''')
s=s.replace('("wire_joint_force", "Wire joint N"),','("wire_joint_force", "Wire joint N"),\n                        ("align_xy_m", "Axis offset m"),\n                        ("tilt_deg", "Axis tilt deg"),\n                        ("ring_lateral_N", "Ring lateral N"),')
a=s.index('    def _on_contacts(');b=s.index('    def _hook_update(',a)
s=s[:a]+'''    def _on_physics_step(self,dt):
        if dt>0:
            self._physics_dt=float(dt)
            self._physics_serial+=1

    def _on_contacts(self,headers,data):
        if self._physics_dt is None:return
        dt=self._physics_dt
        idx=int(self.object_combo.model.get_item_value_model().as_int)
        ring=self.config["paths"]["ring_paths"][idx]
        screw=self.config["paths"]["screws"][idx]
        result={"finger_left_N":0.,"finger_right_N":0.,"ring_axial_N":0.,"ring_lateral_N":0.,"screw_axial_N":0.}
        points=[]
        try:
            for h in headers:
                a=str(PhysicsSchemaTools.intToSdfPath(h.collider0))
                b=str(PhysicsSchemaTools.intToSdfPath(h.collider1))
                pair=a+"|"+b
                finger="panda_leftfinger" in pair or "panda_rightfinger" in pair
                for k in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                    c=data[k]
                    imp=np.asarray(c.impulse,dtype=float)
                    normal=np.asarray(c.normal,dtype=float)
                    force=normal*float(imp)/dt if imp.ndim==0 else imp/dt
                    fn=abs(float(np.dot(force,normal)))
                    if "panda_leftfinger" in pair:result["finger_left_N"]+=fn
                    if "panda_rightfinger" in pair:result["finger_right_N"]+=fn
                    # Target axes are world Z in this fixed assembly; exclude finger squeeze.
                    if ring in pair and not finger:
                        result["ring_axial_N"]+=abs(float(force[2]))
                        result["ring_lateral_N"]+=float(np.linalg.norm(force[:2]))
                    if screw in pair and not finger:result["screw_axial_N"]+=abs(float(force[2]))
                    if finger or ring in pair or screw in pair:
                        points.append({"point":list(c.position),"force_N":force.tolist(),"a":a,"b":b,"physics_dt":dt})
            for key,val in result.items():self._contact_peaks[key]=max(self._contact_peaks.get(key,0.),val)
            self._contact_points=points[:128]
        except Exception as ex:self.user_message="Contact measurement unavailable: "+repr(ex)

''' +s[b:]
s=s.replace('            self.metrics["dt"] = dt','''            self.metrics["dt"] = dt
            idx=int(self.object_combo.model.get_item_value_model().as_int)
            self.config["selected_index"]=idx
            self.config["selected_phase"]=self.fsm.phase
            self.config["paths"]["wire_root"]="/wire"+str(idx+1)
            self.config["paths"]["ring_terminal"]=self.config["paths"]["ring_paths"][idx]
            self.config["paths"]["selected_screw"]=self.config["paths"]["screws"][idx]
            if self._physics_dt is not None and self._physics_serial!=self._last_consumed_serial:
                for key in ["finger_left_N","finger_right_N","ring_axial_N","ring_lateral_N","screw_axial_N"]:
                    self.metrics[key]=self._contact_peaks.get(key,0.)
                self.metrics["grip_force_N"]=max(self.metrics["finger_left_N"],self.metrics["finger_right_N"])
                self.contact_force_N=max(self.metrics["grip_force_N"],self.metrics["ring_axial_N"],self.metrics["screw_axial_N"])
                self._contact_peaks={}
                self._last_consumed_serial=self._physics_serial
            self.metrics["wire_joint_force"]=None
            self.metrics["screw_torque_Nm"]=None
            self.metrics.update(plan_d_runtime.compute_alignment_metrics(self.stage,CALIB,self.config))''')
s=s.replace('            if self.safety.get("stop") and self.controller.armed:','''            if self.config.get("warning_only",False):
                self.safety["would_stop"]=self.safety.get("stop",False)
                self.safety["stop"]=False
                gate["would_halt"]=gate.get("halt",False)
                gate["halt"]=False
            if self.safety.get("stop") and self.controller.armed:''')
s=s.replace('        self._sub_contact = None\n        try:\n            self.event_log.close()', '        self._sub_contact = None\n        self._physics_subscription = None\n        try:\n            self.event_log.close()')
s=s.replace('("STOP: " + ",".join(self.safety.get("latched_reasons") or []) if stop else "MONITORING")','("STOP: " + ",".join(self.safety.get("latched_reasons") or []) if stop else "WARNING ONLY")')
s=s.replace('"safety_stop": self.safety.get("stop"),','"safety_stop": self.safety.get("stop"),\n            "would_stop":self.safety.get("would_stop"),\n            "selected_index":self.config.get("selected_index"),\n            "contact_points":self._contact_points,')
ast.parse(s);update(p,s)

# Promote ONLY after offline mesh, path and USD tests have passed.
audit=json.loads((O/'geometry_audit.json').read_text())
assert audit['checks']['closed_design_no_volume_overlap'] and audit['checks']['helix_path_clear']
assert not audit['checks']['composition_errors']
update(S/'riser_io_preview.usda',(S/'riser_io_printable_v001.usda').read_text())
# Original malformed mother is kept separately even if backups hold a different later copy.
assert (O/'riser_io_preview.original.usda').exists()

doc="""# Task2 printable_v001 — all-printed coarse-thread prototype

NOT M4 / NOT ISO M10. Custom major diameter 10 mm, lead 2 mm, single-start right-hand.
STL coordinates are millimetres. USD coordinates are metres. Do not rescale in the slicer.
No Kit/PhysX/task acceptance has been run. physics_ready=false.

## Print list
| STL | Quantity | Dimensions / role |
|---|---:|---|
| fixture_base.stl | 1 | 154 x 70 mm; body 6 mm, support posts and locating pins |
| terminal_board_with_open_ring_seats.stl | 1 | 140 x 50 mm; 4 mm plate, open seat rims; 28 mm pitch |
| ring_terminal_surrogate.stl | 4 | ring OD 22 mm, bore 11.4 mm, thickness 3 mm; 7.4 mm open-bottom snap cable barrel |
| nut_custom_D10_P2_clearance_030.stl | 4 | 10 mm thick; hex anti-rotation seat; radial thread clearance 0.30 mm |
| screw_custom_D10_P2.stl | 4 + 1 fit test | threaded shaft 18 mm; 17 mm head diameter, 5 mm head height |
| screw_tray_four_real_pockets.stl | 1 | 100 x 100 x 20 mm; 4 true blind pockets, 2 mm floor |
| fit_nut_clearance_020.stl | 1 test | radial clearance 0.20 mm |
| fit_nut_clearance_040.stl | 1 test | radial clearance 0.40 mm |

## First print, not a production specification
0.4 mm nozzle; start with 0.2 mm layer height, >=4 walls. PETG for base/board/tray and screw/nut fit samples.
Ring snap barrel may require PA12 or a locally validated tougher material; do not assume PETG snap life.
Print the screw head-down (rotate 180 degrees and place on bed); protect thread flanks from support scars.
Print the three nut clearances and one screw BEFORE printing all parts. Do not force a tight sample.
The ring barrel matches the authored 7 mm cable diameter. Its retention and the wire2 initial ~4 degree cable tilt remain unverified.
These are mechanical surrogates, not electrically conductive terminals or a working PCB.

## Assembly order
1. Attach base to an independent bench fixture through four 6.6 mm mounting holes. Actual anchoring hardware/clamp is not included; static USD fixation is an explicit boundary condition.
2. Place four printed nuts in the hex pockets. Align their entry ends upwards.
3. Place board on the support posts / registration pins. All four bores are coaxial with nut holes.
4. Snap ring surrogates onto flexible cable ends; verify retention without damaging cable.
5. Seat four rings in their open rim slots, tails through the open side.
6. Pick screws from tray, align and turn clockwise viewed from above while advancing downward.
7. Verify head bears ring, not the rim; test slip/retention only after material and loads are defined.

The simulation begins with rings at cable ends and screws in tray, NOT preinstalled.
The geometric closed pose has ring bottom at fixture Z=18 mm, head underside Z=21 mm and screw tip Z=3 mm.
Coupled helix motion is dz = pitch * dtheta / (2*pi); this is geometry, not a scripted drive.
No kinematic fastening shortcut, fixed-joint switch or direct screw-angle animation is added.
Nine helical sweep poses are volume-clear. Fixed-angle 0.5 mm upward displacement interferes with the nut; this does not prove a rated holding load.

## Open
scene/riser_io_preview.usda (same main entry, backed up)
scene/riser_io_printable_v001.usda (explicit versioned candidate)
Run scene/launch_panel.py only after opening the matching scene.
Choose object index 1..4. Phase1 remains wires, Phase2 screws.
Thresholds are warning-only in manual mode; Esc/STOP remains.
Force channels now read per-contact impulses / physics dt and separate ring from screw.
Torque and cable tension remain unavailable rather than zero.
CAD alignment is calculated from composed USD frames; live tensor/Fabric consistency and GUI control require smoke testing.

## Limits
- Real helical surfaces exist. Dynamic screw/ring use SDF256; static parts use triangle surfaces. Validate SDF cooking and clearances in Kit.
- Closed solid noninterference is NOT friction/preload, thread stripping, material strength or full task validation.
- Original cable layout is inherited. Existing cable/table overlaps and joint dynamics require separate runtime audit.
- Soft wire, TE connector, cage, robot and desk are retained; this pack prints task tooling/terminal/screw surrogates, not the entire robot/workbench/cable harness.
- Success is never inferred from a green color or FSM step selection.
"""
(O/'PRINT_AND_ASSEMBLE.md').write_text(doc)
update(R/'PRINT_BOM.md',doc)
p=R/'TASK_FLOW.md';old=p.read_text()
note="# Printable v001 update (2026-10-24)\n\nPhase1 wires then Phase2 screws is unchanged. The current scene uses all-printed CUSTOM D10/P2 surrogates, not M4. Four seat axes are at the original terminal-row X pitch (28 mm); mounting holes are no longer used as wiring targets. See scene/printable_v001/PRINT_AND_ASSEMBLE.md. Historical M4/stud and missing-proxy statements below are superseded. No physics/task acceptance is claimed.\n\n"
update(p,note+old)
p=S/'LAUNCH.md';update(p,"# Printable v001 scene update\n\nThe same riser_io_preview.usda now contains the custom coarse-thread printable package. Select object index 1–4 in the panel. Threshold policy is warning-only; torque and cable tension are unavailable. Geometry passes offline checks; Kit teleop/fastening not accepted. Print files and preview images: printable_v001/. See PRINT_AND_ASSEMBLE.md.\n\n"+p.read_text())
# Every real edit has a checkpoint diff; outer Git considers this task directory untracked.
patch=[]
for before in sorted(B.iterdir()):
 target=(R/before.name) if before.name in ['PRINT_BOM.md','TASK_FLOW.md'] else S/before.name
 if target.exists():
  patch.extend(difflib.unified_diff(before.read_text().splitlines(True),target.read_text().splitlines(True),fromfile=str(before),tofile=str(target)))
(O/'changes.patch').write_text(''.join(patch))
print('PROMOTED',S/'riser_io_preview.usda')
print('MODIFIED',[str((R/f.name) if f.name in ['PRINT_BOM.md','TASK_FLOW.md'] else S/f.name) for f in B.iterdir()])
