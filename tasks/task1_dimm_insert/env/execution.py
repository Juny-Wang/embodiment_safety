"""Isolated policy execution candidate. No Isaac imports at module load."""
import copy
import math

CONTROL = dict(ik_period_s=.05, ik_position_tolerance_m=.00005,
               ik_rotation_tolerance_rad=.0005, ik_max_evaluations=400)
ACTION = dict(max_goal_lead_m=.005)
class StartupRelease:
    """One-way startup-assist release; reset only with a fresh episode instance."""
    def __init__(self):
        self.released={"left":False,"right":False}
    def update(self,side,angle,press,band,contact_threshold):
        if side not in self.released:raise ValueError(side)
        vals=[press,band,contact_threshold]+([] if angle is None else [angle])
        if not all(math.isfinite(float(x)) for x in vals):
            raise ValueError("Non-finite latch measurement")
        if press>=contact_threshold or (angle is not None and abs(angle)>band):
            self.released[side]=True
        return self.released[side]

def install(panel,emit):
    panel.controller.cfg["control"].update(CONTROL)
    panel.cfg["action"].update(ACTION)
    frozen=copy.deepcopy(dict(control=panel.controller.cfg["control"],
                              action=panel.cfg["action"]))
    release=StartupRelease()
    original=panel.assistance.update_latch_soft_hold
    def update(stage,safety,metrics,calib):
        out=original(stage,safety,metrics,calib)
        for side,item in out["sides"].items():
            if not item.get("ok"):raise RuntimeError("Missing latch joint")
            before=release.released[side]
            released=release.update(side,item["angle_abs_deg"],item["press_N"],
                                    out["band_deg"],out["contact_N"])
            if released:
                joint=stage.GetPrimAtPath(safety["paths"]["socket"]+"/Joints/"+side.title()+"Hinge")
                joint.GetAttribute("drive:angular:physics:stiffness").Set(0.)
                item.update(stiffness=0.,mode="startup_assist_permanently_released")
            if released and not before:
                emit(dict(event="startup_assist_released",time=panel.t,side=side,
                          angle_abs_deg=item["angle_abs_deg"],press_N=item["press_N"]))
        return out
    panel.assistance.update_latch_soft_hold=update
    def check():
        current=dict(control=panel.controller.cfg["control"],action=panel.cfg["action"])
        if current!=frozen:raise RuntimeError("Uniform execution configuration changed")
    return frozen,check
