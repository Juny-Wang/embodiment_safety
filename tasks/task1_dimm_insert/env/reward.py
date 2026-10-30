"""Versioned one-shot stage rewards; evidence is supplied by audited sensors, not commands."""
import math
VERSION = "task1_stage_once_v001"
INTERFACE = "task1_rgb_state31_reward_history_v001"
STAGES = (
 ("both_open", .05, ("insertion_path_clear", "hands_clear", "opening_survives_withdrawal")),
 ("grasp_clear", .10, ("grasp_in_allowed_band", "bilateral_grasp_stable", "dimm_tracks_gripper", "rack_clear")),
 ("insertion_ready", .05, ("key_orientation_correct", "lateral_alignment_ok", "orientation_ok", "insertion_path_clear")),
 ("seated_locked", .20, ("fixed_base_support", "left_hook_engaged", "right_hook_engaged", "no_unintended_penetration", "contact_driven_closure")),
 ("complete", 1., ("released", "retracted", "retention_stable", "independent_acceptance_passed")),
)
class StageReward:
    def __init__(self):
        self.reset()
    def reset(self):
        self.awarded = [False]*5
        self.finished = False
        self.last_time = None
        return self.progress()
    def progress(self):
        return [int(x) for x in self.awarded[:4]]
    def update(self, time, evidence, *, terminated=False, truncated=False, failure=None):
        if self.finished:
            raise RuntimeError("reset required")
        if not math.isfinite(time) or (self.last_time is not None and time <= self.last_time):
            raise ValueError("timestamps must increase")
        if terminated and truncated:
            raise ValueError("termination and truncation are exclusive")
        if failure and not terminated:
            raise ValueError("failure must terminate")
        before=self.progress()
        statuses={}
        fresh=(isinstance(evidence,dict) and evidence.get("valid") is True
            and isinstance(evidence.get("sim_time"),(float,int))
            and math.isfinite(evidence["sim_time"])
            and 0 <= time-evidence["sim_time"] <= 1/240+1e-6
            and bool(evidence.get("source")))
        common=("sensors_valid","no_unintended_penetration","no_safety_violation")
        candidates=[]
        for name,amount,keys in STAGES:
            required=tuple(dict.fromkeys(common+keys))
            if not fresh:
                status="unknown"; missing=list(required)
            else:
                missing=[k for k in required if evidence.get(k) is not True]
                status=("pass" if not missing else "fail" if any(evidence.get(k) is False for k in required) else "unknown")
            statuses[name]={"status":status,"unconfirmed":missing}
            candidates.append(status=="pass")
        # Complete cannot be awarded without current seating/locking evidence.
        candidates[4]=candidates[4] and candidates[3] and terminated and not truncated and not failure
        success=bool(candidates[4])
        if terminated and not failure and not success:
            raise ValueError("terminal requires verified success or failure reason")
        emitted=[]
        # Independent milestones; historical opening is not required for legitimate open initial states.
        if not failure:
            for i,(name,amount,_) in enumerate(STAGES):
                if candidates[i] and not self.awarded[i]:
                    self.awarded[i]=True
                    emitted.append({"stage":name,"reward":amount,"time":time})
        self.last_time=time
        self.finished=bool(terminated or truncated)
        return dict(reward_version=VERSION,interface_version=INTERFACE,
            reward=round(sum(x["reward"] for x in emitted),8),events=emitted,
            progress_before=before,progress_after=self.progress(),current_status=statuses,
            success=success,terminated=bool(terminated),truncated=bool(truncated),
            mask=0. if terminated else 1.,failure=failure,
            evidence_valid=fresh)
