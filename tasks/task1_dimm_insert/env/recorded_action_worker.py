"""Diagnostic recorded-command worker; no policy, simulator or joint writes."""
import hashlib, json, math, os, sys, time, traceback
from pathlib import Path

def validate_rows(rows):
    if not rows:
        raise ValueError("Empty demonstration")
    scales = None
    for i, row in enumerate(rows):
        a = row["actions"]
        if len(a) != 7 or any(not math.isfinite(x) or abs(x)>1.00001 for x in a):
            raise ValueError("Invalid action")
        t = row["observations"]["sim_time"]
        if abs(row["next_observations"]["sim_time"]-t-.1)>1e-6:
            raise ValueError("Recorded interval is not 100ms")
        if i and abs(t-rows[i-1]["next_observations"]["sim_time"])>1e-6:
            raise ValueError("Non-contiguous observations")
        commands = row["diagnostics"]["controller_commands"]
        if len(commands) != 24:
            raise ValueError("Expected 24 physics commands")
        for j,c in enumerate(commands):
            if abs(c["time"]-t-j/240)>1e-6:
                raise ValueError("Physics timing mismatch")
            if any(abs(x-y)>1e-9 for x,y in zip(c["integration_action"],a)):
                raise ValueError("Action varied inside recorded interval")
            if len(c["integration_action"]) != 7:
                raise ValueError("Invalid command shape")
            if abs(c["integration_dt_s"]-1/240)>1e-6:
                raise ValueError("Physics dt mismatch")
            if scales is None:
                scales=c["integration_scales"]
            if scales != c["integration_scales"]:
                raise ValueError("Action scales varied")
    return scales

def main():
    out=Path(sys.argv[1])
    def write(name,value):
        p=out/name;tmp=p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
    try:
        if os.environ.get("TASK1_EVAL")!="1" or os.environ.get("TASK1_ONLINE_LEARN")!="0":
            raise ValueError("Recorded replay is evaluation-only")
        source=Path(os.environ["TASK1_RECORDED_SOURCE"])
        expected=os.environ["TASK1_RECORDED_SHA256"]
        payload=source.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=expected:
            raise ValueError("Source hash mismatch")
        count=int(os.environ["TASK1_MAX_STEPS"])
        rows=[json.loads(s) for s in payload.splitlines()][:count]
        if len(rows)!=count:
            raise ValueError("Insufficient recorded actions")
        scales=validate_rows(rows)
        actual=json.loads((out/"execution_configuration.json").read_text())
        if actual["action"]!=scales:
            raise ValueError("Execution action scales differ from demonstration")
        expected_control=rows[0]["diagnostics"].get("execution_configuration",{}).get("control")
        if expected_control is not None and actual["control"]!=expected_control:
            raise ValueError("Execution control configuration differs from demonstration")
        n=0
        while True:
            req=out/f"request_{n:03d}.json";deadline=time.monotonic()+600
            while not req.exists() and not (out/"policy_stop.json").exists():
                if (out/"actor_error.json").exists():raise RuntimeError("Actor failed")
                if time.monotonic()>deadline:raise TimeoutError("Request")
                time.sleep(.01)
            if not req.exists():break
            if n>=count:raise ValueError("Replay exhausted")
            raw=json.loads((out/f"raw_{n:03d}.json").read_text())
            if n==0:
                ref=rows[0]["observations"]
                distance=lambda a,b:math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
                error=distance(raw["ee_position_m"],ref["ee_position_m"])
                qa,qb=raw["ee_quaternion_xyzw"],ref["ee_quaternion_xyzw"]
                dot=abs(sum(x*y for x,y in zip(qa,qb)))/math.sqrt(sum(x*x for x in qa)*sum(x*x for x in qb))
                angle=math.degrees(2*math.acos(min(1,dot)))
                dimm_error=distance(raw["task_truth"]["poses"]["dimm"][:3],ref["task_truth"]["poses"]["dimm"][:3])
                latch_error=max(abs(raw["task_truth"]["metrics"][s+"_angle_deg"]-ref["task_truth"]["metrics"][s+"_angle_deg"]) for s in ("left","right"))
                write("recorded_replay_contract.json",dict(source=str(source),source_sha256=expected,
                    action_count=count,control_dt_s=.1,physics_hz=240,action_scales_match=True,
                    control_config_present=expected_control is not None,
                    initial_ee_error_m=error,initial_orientation_error_deg=angle,
                    initial_dimm_error_m=dimm_error,initial_latch_error_deg=latch_error,
                    mode="recorded_velocity_commands_only",policy_success_claim=False))
                if error>.0005 or angle>.5 or dimm_error>.0005 or latch_error>.5:
                    raise ValueError("Initial state mismatch")
                start=raw["sim_time"]
            expected_elapsed=rows[n]["observations"]["sim_time"]-rows[0]["observations"]["sim_time"]
            if abs((raw["sim_time"]-start)-expected_elapsed)>1e-6:
                raise ValueError("Live observation timing differs from recorded physical timestamps")
            a=rows[n]["actions"]
            write(f"action_{n:03d}.json",dict(index=n,proposed=a,executed=a,network_updates=0,
                sampling_mode="recorded_demonstration",source_index=rows[n]["index"],
                source_sha256=expected,not_a_learned_policy=True))
            n+=1
        stop=json.loads((out/"policy_stop.json").read_text())
        write("worker_result.json",dict(passed=True,network_updates=0,new_transitions=stop["final_observation_index"],
            learning=False,algorithm="recorded_action_diagnostic",commands_sent=n))
    except BaseException:
        write("worker_error.json",dict(error=traceback.format_exc()))
        raise
if __name__=="__main__":
    main()
