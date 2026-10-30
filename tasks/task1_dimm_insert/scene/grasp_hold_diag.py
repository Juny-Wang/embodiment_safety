"""Grasp-hold slip diagnostic for franka_manual_safety_v002.

After CLOSE (or manual arm), sample finger forces / slip_mm / DIMM speed for ~1.5 s
and classify: hold_ok | friction_slip | impulse_kick | never_secured | ambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math


@dataclass
class GraspHoldDiag:
    duration_s: float = 1.5
    # Classification thresholds (demo / diagnostic, not material limits)
    dual_force_N: float = 0.05
    never_width_m: float = 0.028
    slip_ok_mm: float = 1.5
    slip_friction_mm: float = 2.5
    speed_impulse_mm_s: float = 80.0
    speed_hold_mm_s: float = 25.0
    force_collapse_ratio: float = 0.25

    active: bool = False
    t0: Optional[float] = None
    samples: List[Dict[str, float]] = field(default_factory=list)
    _last_dimm_pos: Optional[Tuple[float, float, float]] = None
    _last_t: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    summary: str = ""

    def arm(self, t: float, reason: str = "close") -> None:
        self.active = True
        self.t0 = float(t)
        self.samples = []
        self._last_dimm_pos = None
        self._last_t = None
        self.result = None
        self.summary = "Grasp diag: sampling %.1fs (%s)..." % (self.duration_s, reason)

    def cancel(self) -> None:
        self.active = False
        self.summary = self.summary or "Grasp diag: cancelled"

    def feed(
        self,
        t: float,
        metrics: Dict[str, Any],
        dimm_pos_m: Optional[Tuple[float, float, float]] = None,
        dt: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.active or self.t0 is None:
            return None
        fl = float(metrics.get("finger_left_N") or 0.0)
        fr = float(metrics.get("finger_right_N") or 0.0)
        slip = metrics.get("slip_mm")
        width = metrics.get("gripper_width_m")
        ee = metrics.get("ee_speed_mm_s")

        dimm_speed = None
        if dimm_pos_m is not None:
            if self._last_dimm_pos is not None and self._last_t is not None:
                dts = float(t) - float(self._last_t)
                if dts > 1e-6:
                    dx = dimm_pos_m[0] - self._last_dimm_pos[0]
                    dy = dimm_pos_m[1] - self._last_dimm_pos[1]
                    dz = dimm_pos_m[2] - self._last_dimm_pos[2]
                    dimm_speed = math.sqrt(dx * dx + dy * dy + dz * dz) / dts * 1000.0
            self._last_dimm_pos = (float(dimm_pos_m[0]), float(dimm_pos_m[1]), float(dimm_pos_m[2]))
            self._last_t = float(t)
        if dimm_speed is None and dt is not None and dimm_pos_m is None:
            pass

        sample = {
            "t": float(t) - float(self.t0),
            "fl": fl,
            "fr": fr,
            "slip_mm": float(slip) if slip is not None else float("nan"),
            "width_m": float(width) if width is not None else float("nan"),
            "ee_mm_s": float(ee) if ee is not None else float("nan"),
            "dimm_mm_s": float(dimm_speed) if dimm_speed is not None else float("nan"),
        }
        self.samples.append(sample)

        if (float(t) - float(self.t0)) < float(self.duration_s):
            n = len(self.samples)
            self.summary = "Grasp diag: %d samples (%.2fs) fl=%.2f fr=%.2f slip=%s" % (
                n,
                sample["t"],
                fl,
                fr,
                "--" if slip is None else "%.2f" % float(slip),
            )
            return None

        self.active = False
        self.result = self._classify(self.samples)
        self.summary = self.result["message"]
        return self.result

    def _finite(self, xs: List[float]) -> List[float]:
        out = []
        for x in xs:
            if x is None:
                continue
            try:
                v = float(x)
            except Exception:
                continue
            if math.isfinite(v):
                out.append(v)
        return out

    def _classify(self, samples: List[Dict[str, float]]) -> Dict[str, Any]:
        if not samples:
            return {
                "verdict": "ambiguous",
                "message": "Grasp diag: no samples",
                "stats": {},
            }
        fls = [s["fl"] for s in samples]
        frs = [s["fr"] for s in samples]
        dual = [1.0 if (s["fl"] >= self.dual_force_N and s["fr"] >= self.dual_force_N) else 0.0 for s in samples]
        dual_frac = sum(dual) / max(1, len(dual))
        slips = self._finite([s["slip_mm"] for s in samples])
        speeds = self._finite([s["dimm_mm_s"] for s in samples])
        widths = self._finite([s["width_m"] for s in samples])
        peak_f = max(max(fls), max(frs))
        end_f = max(fls[-3:] + frs[-3:]) if len(samples) >= 3 else max(fls[-1], frs[-1])
        slip_max = max(slips) if slips else None
        slip_end = slips[-1] if slips else None
        slip_delta = (slips[-1] - slips[0]) if len(slips) >= 2 else None
        speed_max = max(speeds) if speeds else None
        speed_early = max(speeds[: max(1, len(speeds) // 3)]) if speeds else None
        width_min = min(widths) if widths else None

        stats = {
            "n": len(samples),
            "dual_frac": round(dual_frac, 3),
            "peak_finger_N": round(peak_f, 3),
            "end_finger_N": round(end_f, 3),
            "slip_max_mm": None if slip_max is None else round(slip_max, 3),
            "slip_delta_mm": None if slip_delta is None else round(slip_delta, 3),
            "dimm_speed_max_mm_s": None if speed_max is None else round(speed_max, 2),
            "dimm_speed_early_mm_s": None if speed_early is None else round(speed_early, 2),
            "width_min_m": None if width_min is None else round(width_min, 4),
        }

        # 1) Never really secured
        if dual_frac < 0.15 or (width_min is not None and width_min > self.never_width_m and peak_f < 0.5):
            return {
                "verdict": "never_secured",
                "message": (
                    "Grasp diag: NEVER_SECURED — 双指几乎没稳住"
                    " (dual=%.0f%% peak=%.2fN width_min=%s). 更像没夹住，不是后期滑落。"
                    % (100 * dual_frac, peak_f, "--" if width_min is None else "%.1fmm" % (width_min * 1000))
                ),
                "stats": stats,
            }

        # 2) Impulse / artifact kick
        force_collapsed = peak_f >= 1.0 and end_f <= peak_f * self.force_collapse_ratio
        early_kick = speed_early is not None and speed_early >= self.speed_impulse_mm_s
        mid_kick = speed_max is not None and speed_max >= self.speed_impulse_mm_s and force_collapsed
        if early_kick or mid_kick:
            return {
                "verdict": "impulse_kick",
                "message": (
                    "Grasp diag: IMPULSE_KICK — 合拢后 DIMM 速度尖峰"
                    " (vmax=%.0f mm/s early=%.0f peakF=%.2f→endF=%.2f). 偏冲击/薄碰撞假象。"
                    % (
                        speed_max or -1,
                        speed_early or -1,
                        peak_f,
                        end_f,
                    )
                ),
                "stats": stats,
            }

        # 3) Friction / thin-plate slip while still contacting
        if (
            dual_frac >= 0.4
            and slip_delta is not None
            and slip_delta >= self.slip_friction_mm
            and (speed_max is None or speed_max < self.speed_impulse_mm_s * 1.5)
            and end_f >= self.dual_force_N
        ):
            return {
                "verdict": "friction_slip",
                "message": (
                    "Grasp diag: FRICTION_SLIP — 双指力还在，但 slip 变大"
                    " (Δslip=%.2f mm dual=%.0f%% endF=%.2fN). 偏薄板/摩擦不够的正常难夹。"
                    % (slip_delta, 100 * dual_frac, end_f)
                ),
                "stats": stats,
            }

        # 4) Hold OK
        if (
            dual_frac >= 0.5
            and (slip_max is None or slip_max <= self.slip_ok_mm)
            and (speed_max is None or speed_max <= self.speed_hold_mm_s)
            and end_f >= self.dual_force_N
        ):
            return {
                "verdict": "hold_ok",
                "message": (
                    "Grasp diag: HOLD_OK — 这窗里夹住了"
                    " (dual=%.0f%% slip_max=%s dimm_v=%s)."
                    % (
                        100 * dual_frac,
                        "--" if slip_max is None else "%.2fmm" % slip_max,
                        "--" if speed_max is None else "%.0fmm/s" % speed_max,
                    )
                ),
                "stats": stats,
            }

        return {
            "verdict": "ambiguous",
            "message": (
                "Grasp diag: AMBIGUOUS — dual=%.0f%% slip_Δ=%s dimm_v=%s peakF=%.2f. 再夹一次或放宽窗口。"
                % (
                    100 * dual_frac,
                    "--" if slip_delta is None else "%.2fmm" % slip_delta,
                    "--" if speed_max is None else "%.0fmm/s" % speed_max,
                    peak_f,
                )
            ),
            "stats": stats,
        }

