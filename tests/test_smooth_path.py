import unittest
import numpy as np
from tasks.task1_dimm_insert.env.smooth_path import SmoothPath


class SmoothPaths(unittest.TestCase):
    def test_endpoints_and_speed_acceleration(self):
        path = SmoothPath([0, 0, 0], [.2, .1, .3], [.015, -.01, .01], .05)
        start, v0 = path.sample(0)
        end, v1 = path.sample(path.duration)
        np.testing.assert_allclose(start, path.start)
        np.testing.assert_allclose(end, path.end)
        np.testing.assert_allclose(v0, 0, atol=1e-12)
        np.testing.assert_allclose(v1, 0, atol=1e-12)
        times = np.linspace(0, path.duration, 2001)
        velocity = np.array([path.sample(t)[1] for t in times])
        acceleration = np.diff(velocity, axis=0)/(times[1]-times[0])
        self.assertLessEqual(np.linalg.norm(velocity, axis=1).max(), .05001)
        self.assertLessEqual(np.linalg.norm(acceleration, axis=1).max(), .04001)
        np.testing.assert_allclose(acceleration[[0, -1]], 0, atol=.0002)

    def test_geometry_varies_with_shared_endpoints(self):
        a = SmoothPath([0, 0, 0], [.2, 0, 0], [0, .01, .01], .04)
        b = SmoothPath([0, 0, 0], [.2, 0, 0], [0, -.01, .01], .04)
        self.assertGreater(np.linalg.norm(a.sample(a.duration/2)[0]-b.sample(b.duration/2)[0]), .019)
        np.testing.assert_allclose(a.sample(a.duration)[0], b.sample(b.duration)[0])


if __name__ == '__main__':
    unittest.main()
