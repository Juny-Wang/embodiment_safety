import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runners import collect_task1


class CollectionBatch(unittest.TestCase):
    def test_rejected_attempt_replaced_without_exceeding_success_target(self):
        active = 0
        peak = 0
        launched = []
        class Process:
            def __init__(self, command, **kwargs):
                nonlocal active, peak
                self.pid = 1000+len(launched)
                self.polls = 0
                self.done = False
                launched.append(self)
                active += 1
                peak = max(peak, active)
                dest = Path(command[-1])
                dest.mkdir()
                (dest/'result.json').write_text('{}')
            def poll(self):
                nonlocal active
                self.polls += 1
                if self.polls < 2:
                    return None
                if not self.done:
                    active -= 1
                    self.done = True
                return 0
        outcomes = iter([False, True, True, True])
        def assess(*args):
            ok = next(outcomes)
            return dict(accepted=ok, failure_reason=None, reward=1.4 if ok else 0,
                        smoothness=dict(accepted=ok))
        with tempfile.TemporaryDirectory(dir=collect_task1.ROOT/'outputs') as directory:
            out = Path(directory)/'run'
            argv = ['collect', '--output', str(out), '--successes', '3', '--max-attempts', '4', '--workers', '2']
            with patch('sys.argv', argv), patch.object(collect_task1.subprocess, 'Popen', Process), \
                 patch.object(collect_task1, 'assess', assess), patch.object(collect_task1.time, 'sleep'):
                collect_task1.main()
            result = json.loads((out/'summary.json').read_text())
        self.assertEqual(result['status'], 'finished')
        self.assertEqual(result['accepted_count'], 3)
        self.assertEqual(len(result['attempts']), 4)
        self.assertEqual(peak, 2)
        self.assertEqual(active, 0)


if __name__ == '__main__':
    unittest.main()
