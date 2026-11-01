from pathlib import Path
import sys,json,csv,subprocess,shutil
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
run=R/'scene/unsafe_demo_v009'/sys.argv[1];out=R/'scene/unsafe_demo_v009/videos';out.mkdir(exist_ok=True)
report=json.loads((run/'result.json').read_text())
if not report.get('video_frames'):raise RuntimeError('no recorded frames')
rows=list(csv.DictReader((run/'metrics.csv').open()))
start=next(float(r['time_s']) for r in rows if r['phase']=='approach')
sys.path.insert(0,str(R/'scene/single_wire_fastening_v003/video_dependencies'))
import imageio_ffmpeg
ff=imageio_ffmpeg.get_ffmpeg_exe()
dest=out/'task2_unsafe_operation.mp4'
subprocess.run([ff,'-y','-ss',str(max(0,start-.05)),'-i',str(run/'rollout.mp4'),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)
for name in ['hold_bend.png','hold_pull.png','first_exceedance.png']:
 if (run/name).exists():shutil.copy2(run/name,out/name)
shutil.copy2(run/'result.json',out/'measurements.json')
quality='通过接触侵入与关节间隙检查' if report.get('physics_quality_pass') else '物理质量检查存在未通过项，不能作为有效安全评测'
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task2 不安全动作演示</title><style>body{max-width:1280px;margin:32px auto;padding:0 20px;background:#111923;color:#eaf0f6;font:18px/1.7 system-ui}video{width:100%;background:#000}a{color:#8bd4ff}.note{color:#ffc87c}</style><h1>Task2 不安全动作演示</h1><video controls preload="metadata" poster="hold_bend.png"><source src="task2_unsafe_operation.mp4" type="video/mp4"></video><p>机械臂接触抓取 → 转动端子 → 局部过弯 → 回转 → 持续牵拉 → 回退卸载。</p><p class="note">35 mm / 10 N 为暂定演示边界，不是材料损伤极限。护套采用刚性近似；未模拟断裂或绝缘磨损。</p><p>QUALITY</p><p><a href="task2_unsafe_operation.mp4">下载视频</a> · <a href="measurements.json">测量摘要</a></p></html>'''.replace('QUALITY',quality)
(out/'index.html').write_text(html)
print(json.dumps({'video':str(dest),'bytes':dest.stat().st_size,'quality':quality,'report':report},ensure_ascii=False,indent=2))
