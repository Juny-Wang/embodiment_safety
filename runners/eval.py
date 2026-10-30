"""Deterministic complete-task policy evaluation using fresh-process resets."""
import sys,os
from .rollout import main
if __name__=="__main__":
 if "--learn" in sys.argv or "--smoke" in sys.argv:raise ValueError("Evaluation cannot learn or use smoke action wrapper")
 if "--deterministic" not in sys.argv:sys.argv.append("--deterministic")
 os.environ["TASK1_EVAL"]="1"
 main()
