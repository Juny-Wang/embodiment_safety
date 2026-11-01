import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from runners.collect_task1 import smoothness


class CollectionQuality(unittest.TestCase):
    def evaluate(self, jitter):
        rows = []
        for stage in ('LEFT_ABOVE', 'RIGHT_ABOVE', 'GRASP_ABOVE', 'TRANSFER'):
            for i in range(101):
                u = i/100
                rows.append(dict(diagnostics=dict(expert_stage=stage), observations=dict(
                    sim_time=len(rows)*.1, ee_position_m=[.1*u, .01*np.sin(np.pi*u)+jitter*(-1)**i, 0.],
                    ee_quaternion_xyzw=[0., 0., 0., 1.])))
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            (path/'transitions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            return smoothness(path)

    def test_smooth_bowed_path_passes(self):
        quality, samples = self.evaluate(0.)
        self.assertTrue(quality['accepted'])
        self.assertEqual(len(samples), 80)

    def test_fast_oscillation_fails(self):
        quality, _ = self.evaluate(.002)
        self.assertFalse(quality['accepted'])
        self.assertGreater(quality['phases']['TRANSFER']['velocity_jitter_rms_m_s'], .005)


if __name__ == '__main__':
    unittest.main()
