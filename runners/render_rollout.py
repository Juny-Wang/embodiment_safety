"""Offline recorded-RGB GIF; no simulator or invented intermediate frames."""
import argparse,json,math,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from .common import ROOT,resolve
def main():
 p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--mp4",action="store_true");a=p.parse_args()
 run=resolve(a.run).resolve()
 if not run.is_relative_to(ROOT/"outputs"):raise ValueError("Repository outputs only")
 if (run/"replay.gif").exists():raise FileExistsError("Do not overwrite replay")
 result=json.loads((run/"result.json").read_text());episode=Path(result["episode"])
 rows=[json.loads(p.read_text()) for p in sorted(run.glob("raw_*.json"))]
 if not rows:raise ValueError("No recorded observations")
 initial=Rotation.from_quat(rows[0]["ee_quaternion_xyzw"]);start=rows[0]["sim_time"]
 frames=[];durations=[];angles=[];sources={};writer=None
 if a.mp4:
  import imageio_ffmpeg
  if (run/"replay.mp4").exists():raise FileExistsError("Do not overwrite MP4")
  writer=imageio_ffmpeg.write_frames(str(run/"replay.mp4"),(960,330),fps=10,codec="libx264",quality=7,macro_block_size=2)
  writer.send(None)
 for i,row in enumerate(rows):
  angle=math.degrees((Rotation.from_quat(row["ee_quaternion_xyzw"])*initial.inv()).magnitude());angles.append(angle)
  frame=Image.new("RGB",(960,330),"#202020");draw=ImageDraw.Draw(frame)
  for j,key in enumerate(("external","wrist_left","wrist_right")):
   path=episode/row["images"][key]["file"]
   if not path.is_relative_to(episode):raise ValueError("Image path")
   if str(path) not in sources:sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
   with Image.open(path) as im:
    im=im.convert("RGB");im.thumbnail((320,240));frame.paste(im,(320*j,30))
   draw.text((320*j+5,8),key,fill="white")
  draw.text((8,273),f"t={row['sim_time']-start:.3f}s | step={i} | rotation from start={angle:.2f} deg | gripper={row['gripper_width_m']*1000:.1f} mm",fill="white")
  draw.text((8,291),f"Measured EE xyz (m): "+", ".join(f"{x:.4f}" for x in row["ee_position_m"]),fill="white")
  draw.text((8,310),"Recorded RGB, sampled at 10Hz; timestamps are simulation time. Last frame held 1s.",fill="white")
  frames.append(frame)
  if writer:writer.send(frame.tobytes())
  durations.append(max(10,round((rows[i+1]["sim_time"]-row["sim_time"])*100)*10) if i+1<len(rows) else 1000)
 if writer:
  for _ in range(9):writer.send(frames[-1].tobytes())
  writer.close()
 frames[0].save(run/"replay.gif",save_all=True,append_images=frames[1:],duration=[int(x) for x in durations],loop=0)
 report=dict(frames=len(rows),duration_s=rows[-1]["sim_time"]-start,
  max_actual_rotation_from_start_deg=max(angles),final_actual_rotation_from_start_deg=angles[-1],
  initial_position=rows[0]["ee_position_m"],final_position=rows[-1]["ee_position_m"],
  failure_reason=result["failure_reason"],task_success=result["full_task_success"],
  full_transitions=result["full_transitions"],scene_unchanged=result["source_unchanged"],
  source_image_sha256=sources,gif_sha256=hashlib.sha256((run/"replay.gif").read_bytes()).hexdigest(),
  mp4_sha256=hashlib.sha256((run/"replay.mp4").read_bytes()).hexdigest() if a.mp4 else None,
  note="Offline sampled RGB replay, not continuous renderer video; no synthesized frames; final frame held 1s")
 (run/"replay_report.json").write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!="source_image_sha256"},indent=2))
if __name__=="__main__":main()
