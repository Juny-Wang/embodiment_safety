"""Grasp hold diagnostic stub."""
class GraspHoldDiag:
    def __init__(self, *a, **k):
        self.enabled = False
    def feed(self, t, metrics, pos, dt=0.0):
        return {"ok": True, "note": "task2_grasp_diag_stub"}
