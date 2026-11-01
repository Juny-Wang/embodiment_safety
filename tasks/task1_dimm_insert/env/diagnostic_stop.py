"""Optional evaluation-only boundary stop; does not change task reward or success."""
def validate_stage_stop(stage, evaluation, online_learning):
    if stage not in ("", "both_open"):
        raise ValueError("Unsupported diagnostic stop stage")
    if stage and (not evaluation or online_learning):
        raise ValueError("Stage diagnostic stop is evaluation-only")
    return stage

def stage_verified(label, stage):
    if not stage or not label or label.get("failure") or label.get("terminated") or label.get("truncated"):
        return False
    return any(e.get("stage")==stage for e in label.get("events",[]))
