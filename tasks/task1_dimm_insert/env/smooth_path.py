"""Smooth free-space paths with zero endpoint velocity and acceleration."""
import numpy as np


class SmoothPath:
    def __init__(self, start, end, offset, speed, acceleration=.04):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.offset = np.asarray(offset, dtype=float)
        u = np.linspace(0., 1., 201)
        ds = 30*u**2*(1-u)**2
        dds = 60*u - 180*u**2 + 120*u**3
        db = 192*u**2 - 768*u**3 + 960*u**4 - 384*u**5
        ddb = 384*u - 2304*u**2 + 3840*u**3 - 1920*u**4
        delta = self.end-self.start
        peak_v = np.linalg.norm(ds[:, None]*delta + db[:, None]*self.offset, axis=1).max()
        peak_a = np.linalg.norm(dds[:, None]*delta + ddb[:, None]*self.offset, axis=1).max()
        self.duration = max(.5, peak_v/speed, np.sqrt(peak_a/acceleration))

    def sample(self, elapsed):
        u = float(np.clip(elapsed/self.duration, 0., 1.))
        s = 10*u**3 - 15*u**4 + 6*u**5
        ds = 30*u**2*(1-u)**2
        b = 64*u**3*(1-u)**3
        db = 192*u**2 - 768*u**3 + 960*u**4 - 384*u**5
        return (self.start+s*(self.end-self.start)+b*self.offset,
                (ds*(self.end-self.start)+db*self.offset)/self.duration)

    def description(self):
        return dict(start_m=self.start.tolist(), end_m=self.end.tolist(),
                    offset_m=self.offset.tolist(), duration_s=self.duration)
