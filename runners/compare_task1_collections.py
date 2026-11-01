"""Compare measured duration, smoothness, and contact forces by collection seed."""
import argparse
import json
from pathlib import Path

import numpy as np

from .eval_diffusion import write_json


CONTACTS = ('left_press_N', 'right_press_N', 'dimm_environment_N',
            'finger_left_N', 'finger_right_N')


def episode_stats(entry):
    episode = Path(entry['episode'])
    rows = [json.loads(line) for line in (episode/'transitions.jsonl').read_text().splitlines()]
    phases = {}
    for row in rows:
        stage = row['diagnostics']['expert_stage']
        dt = row['next_observations']['sim_time']-row['observations']['sim_time']
        phases[stage] = phases.get(stage, 0.)+dt
    peaks = {k: max(row['diagnostics']['contact_peaks'].get(k, 0.) for row in rows)
             for k in CONTACTS}
    return dict(seed=entry['seed'], episode=str(episode), duration_s=sum(phases.values()),
                phase_duration_s=phases, peak_contact_N=peaks, smoothness=entry['smoothness'])


def compare(baseline, candidate):
    old = json.loads((baseline/'summary.json').read_text())
    new = json.loads((candidate/'summary.json').read_text())
    old_by_seed = {x['seed']: x for x in old['accepted']}
    pairs = [dict(baseline=episode_stats(old_by_seed[x['seed']]), candidate=episode_stats(x))
             for x in new['accepted'] if x['seed'] in old_by_seed]
    all_succeeded = len(new['accepted']) == 3 and len(new['attempts']) == 3
    old_time = float(np.mean([x['baseline']['duration_s'] for x in pairs])) if pairs else None
    new_time = float(np.mean([x['candidate']['duration_s'] for x in pairs])) if pairs else None
    forces = {}
    for key in CONTACTS:
        a = float(np.mean([x['baseline']['peak_contact_N'][key] for x in pairs])) if pairs else None
        b = float(np.mean([x['candidate']['peak_contact_N'][key] for x in pairs])) if pairs else None
        forces[key] = dict(baseline_mean_peak_N=a, candidate_mean_peak_N=b,
                           within_comparison_limit=bool(pairs and b <= 1.2*a+.1))
    gates = dict(three_of_three_successful=all_succeeded, matched_seeds=len(pairs) == 3,
                 at_least_ten_percent_faster=bool(pairs and new_time <= .9*old_time),
                 contact_peaks_comparable=all(v['within_comparison_limit'] for v in forces.values()),
                 smoothness_passed=all(x['smoothness']['accepted'] for x in new['accepted']))
    report = dict(baseline=str(baseline), candidate=str(candidate), pairs=pairs,
                  baseline_mean_duration_s=old_time, candidate_mean_duration_s=new_time,
                  duration_reduction_fraction=1-new_time/old_time if old_time else None,
                  contact_comparison=forces, gates=gates, ready_for_batch=all(gates.values()),
                  scope='Three-seed collection pilot; does not measure BC policy success.')
    write_json(candidate/'speed_comparison.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    a = p.parse_args()
    report = compare(a.baseline, a.candidate)
    print(json.dumps({k: v for k, v in report.items() if k != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
