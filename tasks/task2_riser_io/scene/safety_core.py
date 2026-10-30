"""Shared safety logic for manual control and future RL. No Isaac imports."""
import math
COLORS={"GRAY":(0.55,0.55,0.55,1.),"GREEN":(.15,.85,.3,1.),"YELLOW":(1.,.75,.05,1.),"RED":(1.,.12,.08,1.)}
class SafetyEngine:
 def __init__(self,config):
  self.config=config;self.previous={};self.duration={};self.latched=False;self.reasons=[];self.peaks={}
  for r in config["rules"]:
   if r["limit"] is not None and (not math.isfinite(r["limit"]) or r["limit"]<=0):raise ValueError("Invalid threshold "+r["key"])
 def evaluate(self,metrics,step,dt):
  if not math.isfinite(dt) or dt<=0:raise ValueError("Invalid physics dt")
  rows=[];reasons=[]
  for r in self.config["rules"]:
   k=r["key"];v=metrics.get(k);limit=r["limit"]
   valid=isinstance(v,(int,float)) and math.isfinite(v)
   active=step in r["steps"] and (not r.get("contact_gated") or metrics.get("dimm_contact_N",0)>self.config["contact_gate_N"])
   state="GRAY";ratio=None
   if active and valid and limit is not None:
    ratio=abs(v)/limit
    if ratio>=1:state="RED";reasons.append(k)
    elif ratio>=self.config["warning_ratio"] or (self.previous.get(k)=="YELLOW" and ratio>=self.config["warning_clear_ratio"]):state="YELLOW"
    else:state="GREEN"
   if active and valid:self.peaks[k]=max(self.peaks.get(k,0.),abs(v))
   self.duration[k]=self.duration.get(k,0)+dt if state in ("YELLOW","RED") else 0.
   rows.append(dict(r,value=v if valid else None,color=state,ratio=ratio,peak=self.peaks.get(k),exposure_s=self.duration[k],threshold_status=self.config["status"]))
   self.previous[k]=state
  if metrics.get("sensor_error"):reasons.append("sensor_error")
  if reasons:self.latched=True;self.reasons=list(dict.fromkeys(self.reasons+reasons))
  return {"rows":rows,"stop":self.latched,"current_reasons":reasons,"latched_reasons":list(self.reasons),"task_success":False}
 def acknowledge(self,metrics,step,dt):
  # Never clear while any threshold is still red.
  current=self.evaluate(metrics,step,dt)
  if current["current_reasons"]:return False
  self.latched=False;self.reasons=[];self.peaks={};return True

def observation(metrics):
 """No completion flags inferred from script stage or color."""
 return {k:float(v) if isinstance(v,(int,float)) and math.isfinite(v) else None for k,v in metrics.items()}

def under(path,root):
 return path==root or path.startswith(root+"/")
def dimm_contact_kind(a,b,dimm_root,robot_root):
 if not under(a,dimm_root) and not under(b,dimm_root):return None
 other=b if under(a,dimm_root) else a
 if not under(other,robot_root):return "environment"
 if "panda_leftfinger" in other:return "finger_left"
 if "panda_rightfinger" in other:return "finger_right"
 return "robot_other"
def unexpected_robot_contact(a,b,robot_root,dimm_root,socket_root,step):
 ra,rb=under(a,robot_root),under(b,robot_root)
 if not ra and not rb:return False
 if ra and rb:return True
 robot,other=(a,b) if ra else (b,a)
 if "/panda_link0/" in robot or robot.endswith("/panda_link0"):return False
 finger="panda_leftfinger" in robot or "panda_rightfinger" in robot
 if finger and under(other,dimm_root) and step in range(1,7):return False
 if finger and step==0 and any(under(other,socket_root+"/"+s+"Latch") for s in ["Left","Right"]):return False
 return True
