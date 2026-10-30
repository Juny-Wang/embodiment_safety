"""No-op baseline logger stub for Task2."""
def get_logger(create_if_env=True):
    class _L:
        trial_tag = "task2_noop"
    return _L()
def attach(panel, enable=True, trial_tag=None):
    return None
def maybe_log(panel, dt):
    return None
