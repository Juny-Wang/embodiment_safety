"""Bounded local IK with acceptance based on physical pose tolerances."""
import time

import numpy as np
from scipy.optimize import least_squares


def solve(errors, seed, lower, upper, position_tolerance, rotation_tolerance,
          max_evaluations, method="trf"):
    if method not in ("trf", "dogbox_trf"):
        raise ValueError("Unknown IK solver: " + method)

    def residual(q):
        dp, dr = errors(q)
        return np.r_[dp * 1000., dr * 100.]

    def jacobian(q):
        columns = []
        for k in range(len(q)):
            hi, lo = q.copy(), q.copy()
            hi[k] += 1e-5
            lo[k] -= 1e-5
            columns.append((residual(hi) - residual(lo)) / 2e-5)
        return np.column_stack(columns)

    attempts = []
    candidate = np.asarray(seed, dtype=float)
    for algorithm in (("trf",) if method == "trf" else ("dogbox", "trf")):
        candidate = np.clip(candidate, lower + 1e-8, upper - 1e-8)
        start = time.perf_counter()
        result = least_squares(residual, candidate, bounds=(lower, upper), jac=jacobian,
                               method=algorithm, max_nfev=max_evaluations,
                               xtol=1e-10, ftol=1e-10, gtol=1e-10)
        candidate = result.x
        dp, dr = errors(candidate)
        pe, re = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
        accepted = bool(np.isfinite(candidate).all() and np.all(candidate >= lower) and
                        np.all(candidate <= upper) and pe <= position_tolerance and re <= rotation_tolerance)
        attempts.append(dict(method=algorithm, wall_s=time.perf_counter() - start,
                             nfev=int(result.nfev), solver_success=bool(result.success),
                             accepted=accepted, position_error_m=pe, orientation_error_rad=re))
        if accepted:
            break
    return candidate, dict(attempts=attempts, accepted=accepted,
                           solve_wall_s=sum(x["wall_s"] for x in attempts),
                           local_solver_success=bool(result.success))
