"""Causal shadow evidence reconstruction. No Isaac; never changes source episodes."""
import json, math, hashlib, argparse
from pathlib import Path
from collections import deque,Counter
import numpy as np
from scipy.spatial.transform import Rotation
from .reward import StageReward
LIMITS={"left_press_N":3.,"right_press_N":3.,"dimm_environment_N":5.,"finger_left_N":5.,"finger_right_N":5.}
PARAMS={"hold_s":.3,"grasp_min_N":1.,"contact_gate_N":.05,"rack_rise_m":.055,
        "relative_translation_span_m":.001,"relative_rotation_span_deg":.5,"max_dimm_penetration_m":.0001}
def physics_rows(path):
    with path.open() as f:
        for n,line in enumerate(f,1):
            row=json.loads(line)
            if "sim_time" in row and "contacts" in row:yield n,row
class Evidence:
    def __init__(self,initial):
        self.initial=np.array(initial["task_truth"]["poses"]["dimm"],float)
        self.window=deque()
    def update(self,o,start,frames):
        t=o["sim_time"];times=[r["sim_time"] for _,r in frames]
        valid=bool(o.get("sensor_valid") is True and times and
            abs(times[-1]-t)<=1/240+1e-6 and abs(times[0]-start-1/240)<=1e-6)
        if times:valid=valid and all(abs(b-a-1/240)<=1e-6 for a,b in zip(times,times[1:]))
        peak={k:0. for k in LIMITS};pen=0.;allowed=True;bilateral=True;no_rack=True;support=0.;classified=0
        for _,r in frames:
            m=r.get("metrics",{})
            for k in (*LIMITS,"forbidden_contact","unclassified_grasp_contact"):
                if not isinstance(m.get(k),(int,float)) or not math.isfinite(m[k]) or m[k]<0:valid=False
            for k in LIMITS:
                if isinstance(m.get(k),(int,float)):peak[k]=max(peak[k],m[k])
            bilateral=bilateral and all(m.get("finger_"+side+"_N",-1)>=PARAMS["grasp_min_N"] for side in ("left","right"))
            allowed=allowed and m.get("forbidden_contact")==0 and m.get("unclassified_grasp_contact")==0
            frame_support=0.
            for c in r["contacts"]:
                a,b=c.get("a",""),c.get("b","")
                if "/World/MatchedDIMM/" not in a+b:continue
                if not all(isinstance(c.get(k),(int,float)) and math.isfinite(c[k]) for k in ("separation_m","normal_force_N")):
                    valid=False;continue
                pen=max(pen,-c["separation_m"])
                if c["normal_force_N"]<PARAMS["contact_gate_N"]:continue
                if "panda_leftfinger" in a+b or "panda_rightfinger" in a+b:
                    classified+=1
                    allowed=allowed and c.get("grasp_region")=="ALLOWED"
                if "/World/MatchedRack/" in a+b:no_rack=False
                if "/World/MatchedSocket/Asset/FixedAssembly/" in a+b:
                    frame_support+=c["normal_force_N"]
            support=frame_support
        d=np.array(o["task_truth"]["poses"]["dimm"],float)
        h=np.array(o["ee_position_m"],float);hr=Rotation.from_quat(o["ee_quaternion_xyzw"])
        rel=hr.inv().apply(d[:3]-h);rot=hr.inv()*Rotation.from_quat(d[3:])
        permitted=bool(valid and bilateral and allowed and classified>0)
        if not permitted:self.window.clear()
        self.window.append((t,rel,rot,permitted))
        while len(self.window)>1 and t-self.window[1][0]>=PARAMS["hold_s"]:self.window.popleft()
        duration=t-self.window[0][0]
        span=max(float(np.linalg.norm(x[1]-self.window[0][1])) for x in self.window)
        angle=max(float(np.degrees((x[2]*self.window[0][2].inv()).magnitude())) for x in self.window)
        stable=bool(duration>=PARAMS["hold_s"]-1e-6 and all(x[3] for x in self.window))
        safe=bool(valid and all(peak[k]<=v for k,v in LIMITS.items()) and allowed and
            not any(r.get("controller_error") for _,r in frames))
        e=dict(valid=valid,sim_time=t,source="causal physics.jsonl and measured observation poses",
            sensors_valid=valid,no_safety_violation=safe,
            grasp_in_allowed_band=bool(permitted),
            bilateral_grasp_stable=stable,
            dimm_tracks_gripper=bool(stable and span<=PARAMS["relative_translation_span_m"] and angle<=PARAMS["relative_rotation_span_deg"]),
            rack_clear=bool(valid and no_rack and d[2]-self.initial[2]>=PARAMS["rack_rise_m"]),
            # Only a recorded-contact penetration check, not an exhaustive solid-intersection proof.
            recorded_dimm_contacts_under_penetration_limit=bool(valid and pen<=PARAMS["max_dimm_penetration_m"]),
            fixed_base_contact_force_N=support,
            measurements=dict(relative_span_m=span,relative_span_deg=angle,stable_duration_s=duration,
                rise_m=float(d[2]-self.initial[2]),force_peaks_N=peak,max_contact_penetration_m=pen,
                physics_line_range=[frames[0][0],frames[-1][0]] if frames else None))
        # no_unintended_penetration, swept insertion clearance and hook engagement remain UNKNOWN:
        # absence of penetration in contacts is not proof that no unreported interpenetration exists.
        return e
