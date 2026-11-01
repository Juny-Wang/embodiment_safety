"""Evaluation-only warning policy. Does not alter forces, thresholds or reward evidence."""
import math
def validate_safety_mode(mode, evaluation, learning):
    if mode not in ("stop", "warn"):
        raise ValueError("safety mode must be stop or warn")
    if mode == "warn" and (not evaluation or learning):
        raise ValueError("Warning-only mode is evaluation-only; learning is prohibited")
    return mode

class WarningTracker:
    def __init__(self):
        self.frames = 0
        self.counts = {}
        self.peaks = {}
        self.first_time = None
        self.last_time = None
    def observe(self, sim_time, metrics, limits):
        violations = {}
        for key, limit in limits.items():
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid safety measurement: " + key)
            if value > limit:
                violations[key] = dict(value=value, limit=limit)
        for key in ("forbidden_contact", "unclassified_grasp_contact"):
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid safety measurement: " + key)
            if value > 0:
                violations[key] = dict(value=value, limit=0)
        if not violations:
            return None
        if self.first_time is None:
            self.first_time = sim_time
        self.last_time = sim_time
        self.frames += 1
        for key, item in violations.items():
            self.counts[key] = self.counts.get(key, 0) + 1
            self.peaks[key] = max(self.peaks.get(key, 0), item["value"])
        return dict(sim_time=sim_time, violations=violations, response="record_only")
    def summary(self):
        return dict(violating_physics_frames=self.frames, metric_frame_counts=dict(self.counts),
                    peak_values=dict(self.peaks), first_time=self.first_time, last_time=self.last_time,
                    safety_pass=False if self.frames else None)
