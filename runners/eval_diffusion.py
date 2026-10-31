"""Full Task1 evaluation with a bounded wall-clock budget per episode."""
import os, sys, json, subprocess, hashlib, time, signal
from pathlib import Path
from .common import ROOT, resolve

def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def stop_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit before its Isaac/worker children.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)

def label_summary(dest):
    files = sorted(dest.glob("stage_transition_*.json"),
                   key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    labels = [json.loads(p.read_text()) for p in files]
    return dict(full_transitions=len(labels),
                reward=sum(x["reward"] for x in labels),
                stages=[e["stage"] for x in labels for e in x["events"]])

def timeout_result(dest, wall_s, limit):
    result = dict(completed=True, execution_completed=False,
                  full_task_success=False, wall_timeout=True,
                  failure_reason="wall_time_limit", terminated=False,
                  truncated=True, wall_s=wall_s, wall_timeout_s=limit,
                  outcome="wall_time_limit", **label_summary(dest))
    # This is evaluator bookkeeping, not a fabricated environment transition.
    write_json(dest / "evaluation_timeout.json", result)
    return result

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--wall-timeout-s", type=float, default=1800)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if not 0 < a.wall_timeout_s < float("inf") or a.episodes < 1:
        raise ValueError("Positive finite wall timeout and episode count required")
    cp, out = resolve(a.checkpoint), resolve(a.output)
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Outputs only")
    meta = json.loads(cp.with_suffix(".json").read_text())
    if (meta["algorithm"] != "diffusion_policy" or
        hashlib.sha256(cp.read_bytes()).hexdigest() != meta["sha256"]):
        raise ValueError("Checkpoint")
    if a.resume:
        report = json.loads((out / "summary.json").read_text())
        if report["checkpoint"] != str(cp) or report["requested"] != a.episodes:
            raise ValueError("Resume checkpoint/count mismatch")
        if any(not x.get("completed", False) for x in report["episodes"]):
            raise ValueError("Finalize interrupted episodes before resuming")
    else:
        out.mkdir(parents=True, exist_ok=False)
        report = dict(algorithm="diffusion_policy", checkpoint=str(cp),
                      requested=a.episodes, episodes=[])
    report.update(status="running", wall_timeout_s=a.wall_timeout_s)
    def save():
        report["completed"] = sum(x.get("completed", False) for x in report["episodes"])
        report["successes"] = sum(x.get("full_task_success", False) for x in report["episodes"])
        report["wall_timeouts"] = sum(x.get("wall_timeout", False) for x in report["episodes"])
        write_json(out / "summary.json", report)
    save()
    try:
        done = {x["index"] for x in report["episodes"]}
        for i in range(1, a.episodes + 1):
            if i in done:
                continue
            dest = out / f"episode_{i:02d}"
            if dest.exists():
                raise FileExistsError(dest)
            row = dict(index=i, output=str(dest), completed=False,
                       started_at_unix=time.time(), wall_timeout_s=a.wall_timeout_s)
            report["episodes"].append(row)
            save()
            env = os.environ.copy()
            env.update(TASK1_POLICY_BACKEND="diffusion_policy", TASK1_EVAL="1")
            cmd = [sys.executable, "-B", "-m", "runners.eval",
                   "--checkpoint", str(cp), "--output", str(dest),
                   "--seed", str(20261031 + i)]
            if a.max_steps:
                cmd += ["--max-steps", str(a.max_steps)]
            start = time.monotonic()
            timed_out = False
            with (out / f"episode_{i:02d}.log").open("w") as f:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                row["pid"] = proc.pid
                save()
                try:
                    rc = proc.wait(timeout=a.wall_timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    stop_group(proc)
            if timed_out:
                dest.mkdir(exist_ok=True)
                row.update(timeout_result(dest, time.monotonic()-start, a.wall_timeout_s))
                save()
                continue
            if rc:
                raise RuntimeError("Evaluation process failed: " + str(dest))
            result = json.loads((dest / "result.json").read_text())
            if not result["source_unchanged"] or result["network_updates"] != 0:
                raise RuntimeError("Evaluation contract changed")
            row.update(result, completed=True, wall_s=time.monotonic()-start,
                       wall_timeout=False)
            row.update(label_summary(dest))
            save()
        report["status"] = "finished"
        report["success_rate"] = report["successes"] / a.episodes
    except BaseException as e:
        report["status"] = "error"
        report["error"] = repr(e)
        raise
    finally:
        save()
if __name__ == "__main__":
    main()
