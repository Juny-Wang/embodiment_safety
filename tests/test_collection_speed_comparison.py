import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runners.compare_task1_collections import CONTACTS, compare


class SpeedComparison(unittest.TestCase):
    def run_comparison(self, *, duration=115., force=3.1, seeds=(4, 5, 6), attempts=3):
        def entry(seed, seconds, peak):
            return dict(seed=seed, duration_s=seconds, peak_contact_N={k: peak for k in CONTACTS},
                        smoothness=dict(accepted=True))
        with tempfile.TemporaryDirectory() as temp:
            baseline, candidate = Path(temp)/'baseline', Path(temp)/'candidate'
            baseline.mkdir()
            candidate.mkdir()
            (baseline/'summary.json').write_text(json.dumps(dict(
                accepted=[entry(seed, 140., 3.) for seed in (4, 5, 6)])))
            (candidate/'summary.json').write_text(json.dumps(dict(
                accepted=[entry(seed, duration, force) for seed in seeds], attempts=[{}]*attempts)))
            with patch('runners.compare_task1_collections.episode_stats', side_effect=lambda row: row):
                return compare(baseline, candidate)

    def test_successful_faster_pilot_can_continue(self):
        self.assertTrue(self.run_comparison()['ready_for_batch'])

    def test_speed_does_not_override_force_increase(self):
        report = self.run_comparison(force=4.2)
        self.assertFalse(report['ready_for_batch'])
        self.assertFalse(report['gates']['contact_peaks_comparable'])

    def test_failed_attempt_or_missing_pair_requires_review(self):
        self.assertFalse(self.run_comparison(attempts=4)['ready_for_batch'])
        self.assertFalse(self.run_comparison(seeds=(4, 5, 7))['ready_for_batch'])


if __name__ == '__main__':
    unittest.main()
