"""Supported production entry points. No legacy Task1 PPO/fake-env registration."""
TASKS={"task1_dimm_insert":"tasks.task1_dimm_insert"}
ALGOS={"serl":"algos.serl"}
def list_tasks():return sorted(TASKS)
def list_algos():return sorted(ALGOS)
def resolve_task(name):return TASKS[name]
def resolve_algo(name):return ALGOS[name]
