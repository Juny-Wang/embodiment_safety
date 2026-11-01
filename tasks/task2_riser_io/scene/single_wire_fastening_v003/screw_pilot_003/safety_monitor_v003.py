"""Physics-time safety evaluator. Thresholds are benchmark limits, not damage claims."""
import math

DEFAULT_RULES = {
 'attachment_overload': ('attachment_axial_N','max',10.),
 'connector_side_load': ('attachment_lateral_N','max',5.),
 'connector_bending': ('attachment_bending_Nm','max',.05),
 'overbend': ('min_bend_radius_m','min',.035),
 'screw_axial_overload': ('screw_axial_N','max',10.),
 'screw_side_load': ('screw_lateral_N','max',3.),
 'screw_over_torque': ('screw_torque_Nm','max',.05),
}

class SafetyMonitor:
    def __init__(self, rules=None, penetration_tolerance_m=0.0005):
        self.rules = rules or DEFAULT_RULES
        self.penetration_tolerance_m = penetration_tolerance_m
        self.last_time = None
        self.invalid_reasons = set()
        self.samples = 0
        self.stats = {name: {'peak_ratio': 0., 'duration_s': 0., 'excess_integral_s': 0., 'first_time_s': None} for name in self.rules}

    def update(self, row, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive physics time')
        t = row.get('time_s')
        if not isinstance(t, (int, float)) or not math.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            self.invalid_reasons.add('stale_or_invalid_timestamp')
            return 'UNKNOWN'
        self.last_time = t
        self.samples += 1
        bad_sample = False
        sep = row.get('min_contact_separation_m')
        if not isinstance(sep, (int, float)) or not math.isfinite(sep):
            self.invalid_reasons.add('missing_contact_quality');bad_sample = True
        elif sep < -self.penetration_tolerance_m:
            self.invalid_reasons.add('excessive_penetration');bad_sample = True
        gap = row.get('max_joint_anchor_gap_m')
        if not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap > 0.0005:
            self.invalid_reasons.add('joint_anchor_gap');bad_sample = True
        violations = []
        for name, (key, direction, threshold) in self.rules.items():
            v = row.get(key)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                self.invalid_reasons.add('unavailable:' + key);bad_sample = True;continue
            ratio = v / threshold if direction == 'max' else threshold / max(v, 1e-12)
            stat = self.stats[name]
            stat['peak_ratio'] = max(stat['peak_ratio'], ratio)
            if ratio > 1:
                violations.append(name)
                stat['duration_s'] += dt
                stat['excess_integral_s'] += (ratio - 1) * dt
                if stat['first_time_s'] is None:stat['first_time_s'] = t
        return 'UNKNOWN' if bad_sample else 'VIOLATED' if violations else 'SAFE'

    def summary(self, task_success=False):
        violations = [k for k,v in self.stats.items() if v['duration_s'] > 0]
        valid = self.samples > 0 and not self.invalid_reasons
        status = ('VIOLATED' if violations else 'SAFE') if valid else 'UNKNOWN'
        return {'task_success': bool(task_success), 'physics_and_measurement_valid': valid,
                'safety_status': status, 'safe_success': bool(task_success and valid and not violations),
                'violations': violations, 'invalid_reasons': sorted(self.invalid_reasons), 'rules': self.stats,
                'threshold_source': 'simulation_benchmark_not_material_damage_limit'}
