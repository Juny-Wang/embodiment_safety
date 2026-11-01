"""Re-solve recorded Task1 IK failures without starting Isaac or changing training."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Training directory containing episode_NNN')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--isaac-root', type=Path, default=Path('/mnt/data2/isaacsim-5.0.0'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.isaac_root / 'exts/isaacsim.robot_motion.lula/pip_prebundle'))
    import lula

    policies = args.isaac_root / 'exts/isaacsim.robot_motion.motion_generation/motion_policy_configs'
    policy_map = json.loads((policies / 'policy_map.json').read_text())
    config_path = policies / policy_map['Franka']['RMPflow']
    paths = json.loads(config_path.read_text())['relative_asset_paths']
    description = (config_path.parent / paths['robot_description_path']).resolve()
    urdf = (config_path.parent / paths['urdf_path']).resolve()
    robot = lula.load_robot(str(description), str(urdf))
    kinematics = robot.kinematics()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(input=str(args.input.resolve()), robot_description=str(description), urdf=str(urdf),
                  python=sys.version, numpy=np.__version__, scipy=scipy.__version__, lula=lula.__file__,
                  tolerances_changed=False, simulator_started=False, cases=[])

    for episode in range(1, args.episodes + 1):
        source = args.input / f'episode_{episode:03d}' / 'controller_diagnostics.json'
        saved = json.loads(source.read_text())
        diagnostic, control = saved['ik'], saved['control']
        frame = control['ee_frame']
        measured = np.array(diagnostic['measured_joint_positions'])
        failed = np.array(diagnostic['candidate_joint_positions'])
        lower, upper = np.array(diagnostic['lower_limits_rad']), np.array(diagnostic['upper_limits_rad'])
        target_p = np.array(diagnostic['commanded_position_m'])
        target_R = Rotation.from_quat(diagnostic['commanded_quaternion_xyzw'])
        pos_tol, rot_tol = diagnostic['position_tolerance_m'], diagnostic['orientation_tolerance_rad']

        # Reconstruct the base transform from the recorded measured joint/EE pair.
        # The independent failed candidate must then reproduce both logged errors.
        local = kinematics.pose(measured.reshape(-1, 1), frame)
        base_R = Rotation.from_quat(diagnostic['measured_ee_quaternion_xyzw']).as_matrix() @ local.rotation.matrix().T
        base_p = np.array(diagnostic['measured_ee_position_m']) - base_R @ local.translation.reshape(3)

        def errors(q):
            pose = kinematics.pose(np.asarray(q).reshape(-1, 1), frame)
            p = base_R @ pose.translation.reshape(3) + base_p
            R = base_R @ pose.rotation.matrix()
            return p - target_p, (Rotation.from_matrix(R) * target_R.inv()).as_rotvec()

        def residual(q):
            dp, dr = errors(q)
            return np.r_[dp * 1000., dr * 100.]

        def jacobian(q):
            columns = []
            for j in range(len(q)):
                hi, lo = q.copy(), q.copy()
                hi[j] += 1e-5
                lo[j] -= 1e-5
                columns.append((residual(hi) - residual(lo)) / 2e-5)
            return np.column_stack(columns)

        def assess(q):
            dp, dr = errors(q)
            pe, re = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
            margin = float(np.min(np.minimum(q - lower, upper - q)))
            return dict(position_error_m=pe, orientation_error_rad=re,
                        within_tolerance=bool(pe <= pos_tol and re <= rot_tol and margin >= 0 and np.isfinite(q).all()),
                        min_joint_margin_rad=margin, max_joint_delta_from_measured_rad=float(np.max(np.abs(q-measured))),
                        joint_positions=np.asarray(q).tolist())

        reproduced = assess(failed)
        for field in ('position_error_m', 'orientation_error_rad'):
            if abs(reproduced[field] - diagnostic[field]) > 1e-8:
                raise ValueError(f'{source}: reconstructed FK disagrees with logged {field}')
        singular_values = np.linalg.svd(jacobian(failed), compute_uv=False)
        case = dict(episode=episode, source=str(source.resolve()), frame=frame,
                    position_tolerance_m=pos_tol, orientation_tolerance_rad=rot_tol,
                    reconstructed_base_position_m=base_p.tolist(), reconstructed_base_rotation=base_R.tolist(),
                    logged_failure=reproduced, weighted_jacobian_singular_values=singular_values.tolist(),
                    previous_successful_ik_seed_available=False, trials=[])

        for method in ('trf', 'dogbox'):
            for seed_name, seed in (('measured', measured), ('failed_candidate', failed)):
                start = time.perf_counter()
                result = least_squares(residual, np.clip(seed, lower+1e-8, upper-1e-8),
                    bounds=(lower, upper), jac=jacobian, method=method,
                    max_nfev=control['ik_max_evaluations'], xtol=1e-10, ftol=1e-10, gtol=1e-10)
                elapsed = time.perf_counter()-start
                trial = dict(method='scipy_'+method, seed=seed_name, wall_s=elapsed,
                             solver_success=bool(result.success), status=int(result.status),
                             nfev=result.nfev, message=str(result.message), **assess(result.x))
                case['trials'].append(trial)
                print(json.dumps(dict(episode=episode, **trial)), flush=True)

        target_local_p = base_R.T @ (target_p-base_p)
        target_local_R = base_R.T @ target_R.as_matrix()
        pose = lula.Pose3(lula.Rotation3(target_local_R), target_local_p.reshape(3, 1))
        for seed_name, seed in (('measured', measured), ('failed_candidate', failed)):
            cfg = lula.CyclicCoordDescentIkConfig()
            cfg.position_tolerance = pos_tol
            cfg.orientation_tolerance = 2*np.sin(rot_tol/2)
            cfg.cspace_seeds = [seed]
            cfg.max_num_descents = 1
            cfg.sampling_seed = 20261101
            start = time.perf_counter()
            result = lula.compute_ik_ccd(kinematics, pose, frame, cfg)
            elapsed = time.perf_counter()-start
            q = np.asarray(result.cspace_position).reshape(-1)
            trial = dict(method='lula_ccd_bfgs', seed=seed_name, wall_s=elapsed,
                         solver_success=bool(result.success), max_num_descents=1, **assess(q))
            case['trials'].append(trial)
            print(json.dumps(dict(episode=episode, **trial)), flush=True)
        report['cases'].append(case)
        (args.output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))

    lines = ['# Task1 offline IK re-solve', '',
             'All candidates use the recorded targets, joint limits and original tolerances. '
             'The logged failed-candidate errors are reproduced before each comparison. '
             'Lula is limited to one supplied seed, with no additional random restarts. '
             'Acceptance uses the original rotation-vector norm, regardless of solver success flags.', '',
             '| Episode | Method | Seed | Accepted | Position error (mm) | Rotation error (deg) | Solve time (ms) | Max joint change from measured (deg) |',
             '|---|---|---|---|---|---|---|---|']
    for case in report['cases']:
        for trial in case['trials']:
            lines.append(f"| {case['episode']} | {trial['method']} | {trial['seed']} | {trial['within_tolerance']} | "
                         f"{trial['position_error_m']*1000:.6f} | {np.rad2deg(trial['orientation_error_rad']):.6f} | "
                         f"{trial['wall_s']*1000:.3f} | {np.rad2deg(trial['max_joint_delta_from_measured_rad']):.3f} |")
    lines += ['', '## Limits', '',
              'These are five frozen failure snapshots, not closed-loop rollouts or a performance benchmark. '
              'The last successful IK solution was not recorded; a failed-candidate restart is additional '
              'optimization, not a replay of the original solve. Valid kinematic solutions do not establish '
              'collision-free motion, physical tracking or assembly success. No training or controller settings were changed.', '',
              '## Reproduce', '', '```bash',
              f"{args.isaac_root}/python.sh runners/analyze_ik_failures.py --input {args.input} --output <new-output-directory>",
              '```', '']
    (args.output / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
