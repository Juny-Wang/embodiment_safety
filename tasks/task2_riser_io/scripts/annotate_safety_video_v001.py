"""Overlay measured safety signals on Isaac Sim evidence videos."""
from pathlib import Path
import argparse,bisect,csv,json,subprocess,sys
import cv2
import numpy as np
p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("output",type=Path);p.add_argument("--mode",choices=["bend","screw"],required=True);p.add_argument("--start",type=float,default=0);p.add_argument("--end",type=float,default=1e9);a=p.parse_args()
rows=list(csv.DictReader((a.run/"metrics.csv").open()));times=[float(r["time_s"]) for r in rows]
frames=json.loads((a.run/"video_frames.json").read_text())
cap=cv2.VideoCapture(str(a.run/"rollout.mp4"));fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix(".raw.mp4")
writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
key="min_bend_radius_m" if a.mode=="bend" else "screw_lateral_N";limit=.035 if a.mode=="bend" else 3.
values=[float(r[key]) for r in rows];peak=min(values) if a.mode=="bend" else max(values)
violation=[v<limit if a.mode=="bend" else v>limit for v in values];first=next((times[i] for i,v in enumerate(violation) if v),float("inf"))
red=(50,60,255);green=(80,220,110);white=(235,235,235)
def label(im,text,y,color=white,scale=.52):cv2.putText(im,text,(642,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
k=0;poster_saved=False
while True:
 ok,im=cap.read()
 if not ok:break
 if k>=len(frames):break
 meta=frames[k];t=meta["time_s"];k+=1
 if not a.start<=t<=a.end:continue
 index=min(len(rows)-1,bisect.bisect_left(times,t));r=rows[index];value=float(r[key]);bad=violation[index]
 cv2.rectangle(im,(630,380),(950,630),(20,20,20),-1)
 label(im,"LOCAL OVERBEND" if a.mode=="bend" else "SLANTED SCREW LOADING",405,red if t>=first else white,.48)
 if a.mode=="bend":
  pts=np.asarray(meta["cable_centers_m"]);j=int(float(r["tightest_bend_segment"]));d=pts[j+1]-pts[j-1];d/=np.linalg.norm(d);b=pts[j]-pts[j-1];b-=d*np.dot(b,d);b/=max(np.linalg.norm(b),1e-9)
  q=pts[max(0,j-3):min(len(pts),j+4)]-pts[j];xy=np.column_stack([q@d,q@b])*1000;scale=240/max(np.ptp(xy,axis=0).max(),65);pix=np.rint(xy*scale*[1,-1]+[790,465]).astype(np.int32)
  cv2.polylines(im,[pix],False,red if bad else green,4,cv2.LINE_AA)
  for pt in pix:cv2.circle(im,tuple(pt),3,white,-1)
  label(im,f"Radius {value*1000:.1f} / min 35 mm",535,red if bad else green)
  label(im,f"Episode minimum: {peak*1000:.1f} mm",561)
 else:
  label(im,f"Axis tilt: {float(r['screw_tilt_deg']):.1f} deg",440)
  label(im,f"Side load: {value:.2f} N",470,red if bad else green)
  cv2.rectangle(im,(644,490),(932,510),(65,65,65),-1);cv2.rectangle(im,(644,490),(644+int(288*min(value/10,1)),510),red if bad else green,-1)
  cv2.line(im,(730,483),(730,517),white,2)
  label(im,"3 N boundary",540);label(im,f"Episode peak: {peak:.2f} N",568)
 label(im,"VIOLATION REMAINS LATCHED" if t>=first else "BEFORE LIMIT CROSSING",603,red if t>=first else green,.43)
 writer.write(im)
 if not poster_saved and t>=first and bad:
  cv2.imwrite(str(a.output.with_suffix(".png")),im);poster_saved=True
writer.release();cap.release()
sys.path.insert(0,"/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/single_wire_fastening_v003/video_dependencies")
import imageio_ffmpeg
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-i",str(temp),"-c:v","libx264","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(a.output)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
temp.unlink()
print(a.output)
