"""Daily 23:30 code sync for the Embodied_Safety repository only."""
import os,sys,subprocess,fcntl
from pathlib import Path
from datetime import datetime
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
ENV=dict(os.environ,GIT_TERMINAL_PROMPT="0",GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=20")
def run(*args):
    p=subprocess.run(args,cwd=ROOT,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    if p.stdout: print(p.stdout,flush=True)
    if p.returncode: raise RuntimeError("Command failed: "+str(args[:3])+" exit="+str(p.returncode))
    return p.stdout.strip()
def check():
    assert run("git","rev-parse","--show-toplevel")==str(ROOT)
    assert run("git","branch","--show-current")=="main"
    assert run("git","remote","get-url","origin")=="git@github.com-juny:Juny-Wang/embodiment_safety.git"
    run("git","status","--short")
if "--check" in sys.argv:
    check()
    run("git","ls-remote","origin","refs/heads/main")
    print("Read-only preflight passed; no commit or push.")
    raise SystemExit(0)
out=ROOT/"outputs/daily_sync";out.mkdir(parents=True,exist_ok=True)
with (out/"sync.lock").open("a") as lock:
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit("Another daily sync is running")
    log=(out/(datetime.now().strftime("%Y-%m-%d")+".log")).open("a",buffering=1)
    sys.stdout=log;sys.stderr=log
    print("\nSTART",datetime.now().isoformat(),flush=True)
    try:
        check()
        run("git","add","-A")
        run("git","status","--short")
        staged=run("git","diff","--cached","--name-only")
        if staged:
            run("git","diff","--cached","--stat")
            run("git","commit","-m","Daily code update "+datetime.now().strftime("%Y-%m-%d"))
        else: print("No changes; no new commit.",flush=True)
        run("git","push","origin","main")
        local=run("git","rev-parse","HEAD")
        remote=run("git","ls-remote","origin","refs/heads/main").split()[0]
        assert local==remote,"Remote main differs from local HEAD"
        print("SUCCESS",local,flush=True)
    except Exception as exc:
        print("FAILED",repr(exc),"No force push or automatic merge; retry at next scheduled run.",flush=True)
        raise SystemExit(1)
