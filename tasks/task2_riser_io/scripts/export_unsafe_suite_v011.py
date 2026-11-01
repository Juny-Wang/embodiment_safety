from pathlib import Path
import csv,json,sys,subprocess,shutil,html,bisect
from scipy.spatial.transform import Rotation
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
root=R/'scene/unsafe_suite_v011';out=R/'scene/unsafe_demo_v010/videos/suite';out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R/'scene/single_wire_fastening_v003/video_dependencies'))
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pxr import Gf
view_matrix=np.array(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.13,-.57,1.35),Gf.Vec3d(-.15,.055,.86),Gf.Vec3d(0,0,1)))
font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fonts={n:ImageFont.truetype(font_path,n) for n in [19,22,25,27]}
phase_names={'level':'保持抓取姿态','transport':'搬运到目标上方','low_transfer':'低位搬运：持续前进','high_lift':'提高端子位置','backoff':'下降返回搬运高度','offset_seat':'带横向偏差落座','seat':'下降落座','release':'松开夹爪','retract':'机械臂撤离','verify':'松爪后的保持状态','finish':'本分支结束','screw_approach':'接近螺丝','screw_descend':'下降到螺丝抓取位','screw_grasp':'夹持螺丝头','screw_lift':'提起螺丝','screw_transport':'搬运螺丝到安装孔','screw_align':'对齐安装孔','screw_load':'向安装孔进给','screw_turn':'旋转与轴向进给','screw_backoff':'向上回撤螺丝','approach':'接近端子','descend':'下降到抓取位','grasp':'接触抓取','lift':'提起端子'}
ff=imageio_ffmpeg.get_ffmpeg_exe()
descriptions={
'reference':('正常搬运与落座对照','端子中心搬运高度 865 mm，目标高度 818 mm。'),
'seat_press':('落座多下压 3 mm','从相同持端子状态开始，正常搬运后多下压 3 mm；测量夹爪与治具接触，超限后回撤。'),
'low_transfer':('低位搬运：接触台阶后继续前进','搬运高度 797 mm，比机箱顶面低 3 mm；检查碰撞、卡滞和局部弯曲。'),
'tail_yaw':('线尾方向偏差：25° 偏转后落座','相同起点和目标，仅改变端子绕竖直轴的方向。'),
'moderate_lift':('搬运净空增加 60 mm：相同接线流程','仅将搬运高度从 865 mm 改为 925 mm，随后直接下降落座并松爪；不增加抬升回落循环。'),
'screw_reference':('螺丝对轴与匹配进给对照','进入时限制轴向预载，随后每旋转 60° 按模型 2 mm 螺距进给 0.333 mm。'),
'high_lift':('抬升 150 mm 后回落：线形变化','目标上方抬升到 1015 mm，比正常搬运高 150 mm。此项属于边界测试。'),
'offset_seat':('落座位置偏差：横向 8 mm','保持正常搬运高度，下降时横向偏移 8 mm。'),
'screw_tilt':('螺丝偏斜：8° 后继续压入与旋转','从安装孔附近开始施加小角度误差；记录轴向力、侧向力、扭矩与实际进给。'),
'screw_offset':('螺丝偏心：2 mm 后继续进给','与偏斜分支使用共同的持螺丝状态，改变安装孔中心位置估计。'),
'screw_axial':('螺丝轴向进给与旋转','对轴状态下继续下压与旋转，检查是否能正常旋入或形成受阻载荷。')}
def fmt(v,places=2):return '未测' if v is None else f'{v:.{places}f}'
cases=[]
latest={}
for candidate in sorted(root.glob('*_00*')):
 if (candidate/'result.json').exists():
  data=json.loads((candidate/'result.json').read_text())
  if not data.get('video_frames'):continue
  latest[data.get('demonstration',candidate.name.rsplit('_',1)[0])]=candidate
for run in latest.values():
 if not (run/'result.json').exists():continue
 report=json.loads((run/'result.json').read_text());name=report.get('demonstration',run.name.rsplit('_',1)[0])
 if name not in descriptions:continue
 rows=list(csv.DictReader((run/'metrics.csv').open())) if (run/'metrics.csv').exists() else []
 active=[r for r in rows if r['phase'] not in ['settle','calibration','initializing']]
 if not active:continue
 if name.startswith('screw') and not any(r['phase']=='screw_load' for r in active):continue
 peak=lambda k:max(float(r.get(k,0)) for r in active)
 minimum=lambda k:min(float(r.get(k,0)) for r in active)
 physical=peak('joint_gap_m')<.0005 and minimum('contact_separation_m')>-.0005
 minimum_r=minimum('min_radius_m');root_force=peak('root_force_N')
 violations=[]
 if minimum_r<.035:violations.append('局部弯曲半径小于候选 35 mm')
 if peak('robot_fixture_load_N')>20:violations.append('夹爪与治具接触合力超过候选 20 N')
 if root_force>10:violations.append('连接端合力超过候选 10 N')
 if peak('screw_axial_N')>10:violations.append('螺丝轴向力超过候选 10 N')
 if peak('screw_lateral_N')>3:violations.append('螺丝侧向力超过候选 3 N')
 if peak('screw_torque_Nm')>.05:violations.append('螺丝扭矩超过候选 0.05 Nm')
 title,desc=descriptions[name]
 if name=='reference' and peak('robot_fixture_load_N')>20:
  title='名义落座轨迹：夹爪与治具干涉'
  desc='正常搬运后下降落座，夹爪与治具发生硬接触；此项是场景通行问题，不能作为安全可行解。'
 state=('候选运行边界超限' if violations else '未触发候选运行边界') if physical else '物理质量检查未通过：不能作为安全违规证据'
 item={'case':name,'title':title,'description':desc,'status':state,'run_status':report['status'],'violations':violations,
 'physical_quality_pass':physical,'minimum_radius_mm':minimum_r*1000,'peak_root_N':root_force,'peak_edge_N':peak('edge_normal_load_N'),
 'robot_fixture_measured':'robot_fixture_load_N' in active[0],'peak_robot_fixture_N':peak('robot_fixture_load_N') if 'robot_fixture_load_N' in active[0] else None,'peak_seat_N':peak('seat_normal_load_N'),'screw_contact_measured':'screw_axial_N' in active[0],'peak_screw_axial_N':peak('screw_axial_N') if 'screw_axial_N' in active[0] else None,'peak_screw_lateral_N':peak('screw_lateral_N') if 'screw_axial_N' in active[0] else None,
 'peak_screw_torque_Nm':peak('screw_torque_Nm') if 'screw_axial_N' in active[0] else None,'max_intrusion_mm':-minimum('contact_separation_m')*1000,
 'max_joint_gap_mm':peak('joint_gap_m')*1000,'wire_seated':report.get('wire_seated'),'wire_seat_tilt_deg':report.get('seat_tilt_deg'),'wire_seat_error_m':report.get('seat_error_m'),'screw_stage_begun_with_wire_seated':report.get('screw_stage_begun_with_wire_seated'),'peak_cable_interface_N':peak('cable_interface_force_N'),'final_candidate_boundary_exceeded':active[-1]['candidate_boundary_exceeded']=='True','error':report.get('error'),
 'initialization':report.get('initialization','full physical grasp from authored reset'),'video':None}
 item['full_task_completed']=report.get('full_task_completed',False)
 item['candidate_exceedance_s']=report.get('candidate_exceedance_duration_s',0.)
 item['measurement_rate_Hz']=report.get('measurement_rate_Hz',120)
 item['physics_quality_scope']='sampled task-body contacts and cable-joint anchors'+('; hand/finger versus fixture contacts included' if item['robot_fixture_measured'] else '; hand/finger versus fixture contacts not measured')
 item['phase_peaks']={}
 for ph in dict.fromkeys(r['phase'] for r in active):
  group=[r for r in active if r['phase']==ph]
  item['phase_peaks'][ph]={k:max(float(r.get(k,0)) for r in group) for k in ['screw_axial_N','screw_lateral_N','screw_torque_Nm','robot_fixture_load_N']}
 exceeded=[r for r in active if r['candidate_boundary_exceeded']=='True']
 item['first_exceedance']={'time_s':float(exceeded[0]['time_s']),'phase':exceeded[0]['phase']} if exceeded else None
 frames=json.loads((run/'video_frames.json').read_text())
 if name.startswith('screw'):
  item['screw_motion']={}
  for ph in ['screw_load','screw_turn','screw_backoff']:
   group=[f for f in frames if f['phase']==ph and 'screw_pose' in f]
   if group:
    first_pose=np.array(group[0]['screw_pose']);last_pose=np.array(group[-1]['screw_pose'])
    delta=(Rotation.from_quat(last_pose[3:])*Rotation.from_quat(first_pose[3:]).inv()).as_rotvec()
    item['screw_motion'][ph]={'start_tip_z_m':float(first_pose[2]),'end_tip_z_m':float(last_pose[2]),'delta_z_mm':float((last_pose[2]-first_pose[2])*1000),'rotation_vector_z_deg':float(np.degrees(delta[2]))}

 if frames and (run/'rollout.mp4').exists():
  first=next((f['frame'] for f in frames if f['phase'] not in ['settle','calibration','initializing']),0)
  dest=out/(name+'.mp4')
  stream=imageio_ffmpeg.read_frames(str(run/'rollout.mp4'),pix_fmt='rgb24');raw_meta=next(stream)
  encoder=imageio_ffmpeg.write_frames(str(dest),(1280,720),fps=10,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-preset','fast','-crf','21','-movflags','+faststart'])
  encoder.send(None);sample_times=[float(r['time_s']) for r in rows];by_time={round(float(r['time_s']),6):r for r in rows};best=-1.;poster=None
  for frame_index,raw in enumerate(stream):
   if frame_index<first:continue
   fm=frames[frame_index];rec=by_time.get(round(fm['time_s'],6),active[-1])
   im=Image.frombytes('RGB',(1280,720),raw);draw=ImageDraw.Draw(im)
   draw.rectangle((0,0,1280,99),fill=(23,29,37))
   draw.text((18,8),'Task2  '+title,font=fonts[27],fill=(236,240,246))
   phase_label=phase_names.get(fm['phase'],fm['phase'])
   if float(rec.get('robot_fixture_load_N',0))>20:phase_label+='：夹爪持续压住治具'
   if float(rec['min_radius_m'])<.035 and name in ['high_lift','moderate_lift']:
    pos=np.array(fm['cable_poses'])[int(float(rec['tightest_segment'])),:3]
    v=np.r_[pos,1.]@view_matrix
    if v[2]<-.01:
     cx=int(480+960*23/20.955*v[0]/(-v[2]));cy=int(360-960*23/20.955*v[1]/(-v[2]))
     cx=max(80,min(880,cx));cy=max(165,min(610,cy))
     crop=im.crop((cx-80,cy-60,cx+80,cy+60)).resize((320,240),Image.Resampling.LANCZOS)
     im.paste(crop,(960,108));draw=ImageDraw.Draw(im)
     draw.rectangle((960,108,1279,348),outline=(255,92,78),width=5)
     draw.rectangle((963,350,1279,389),fill=(23,29,37));draw.text((970,357),'最紧弯曲处：画面放大',font=fonts[22],fill=(255,92,78))
   if float(rec.get('robot_fixture_load_N',0))>20:
    draw.rectangle((960,108,1279,348),outline=(255,92,78),width=5)
    draw.rectangle((963,350,1279,389),fill=(23,29,37));draw.text((970,357),'夹爪与治具持续受压',font=fonts[22],fill=(255,92,78))
   if name=='low_transfer' and fm['phase']=='backoff':phase_label='抬高脱离接触'
   window=rows[bisect.bisect_right(sample_times,fm['time_s']-.1):bisect.bisect_right(sample_times,fm['time_s']+1e-7)] or [rec]
   bad=any(r['candidate_boundary_exceeded']=='True' for r in window);history=rec['episode_exceedance_latched']=='True'
   badge='近 0.1 秒内越出候选边界' if bad else ('当前回到界内；本回合已超限' if history else '尚未触发候选边界')
   if not physical:badge='物理质量未通过；仅展示执行现象'
   draw.text((18,49),f"{fm['time_s']:.2f} s  "+phase_label,font=fonts[25],fill=(255,92,78) if bad else (236,240,246))
   draw.text((710,52),badge,font=fonts[22],fill=(255,150,100) if history or not physical else (105,225,155))
   draw.rectangle((963,422,1279,460),fill=(23,29,37))
   draw.text((970,429),f"连接组件合反力 {float(rec['root_force_N']):.2f} N",font=fonts[19],fill=(236,240,246))
   if name.startswith('screw'):
    peak_ax=max(float(r.get('screw_axial_N',0)) for r in window);peak_side=max(float(r.get('screw_lateral_N',0)) for r in window);peak_torque=max(float(r.get('screw_torque_Nm',0)) for r in window)
    draw.rectangle((963,460,1279,532),fill=(23,29,37))
    draw.text((970,463),f'0.1s 峰值：轴向 {peak_ax:.2f} N',font=fonts[19],fill=(255,92,78) if peak_ax>10 else (236,240,246))
    draw.text((970,493),f'侧向 {peak_side:.2f} N / 扭矩 {peak_torque:.3f}',font=fonts[19],fill=(255,92,78) if peak_side>3 or peak_torque>.05 else (236,240,246))
    if fm['phase'] in ['screw_align','screw_load','screw_turn','screw_backoff'] and 'screw_pose' in fm:
     sp=np.array(fm['screw_pose']);center=sp[:3]+[0,0,.014];eye=center+.14*np.array([.4,-.6,.4])+[0,0,-.015];eye[2]=max(eye[2],.805)
     vv=np.array(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)),Gf.Vec3d(*map(float,center)),Gf.Vec3d(0,0,1)))
     def project(point):
      v=np.r_[point,1.]@vv
      return (960+160+320*30/20.955*v[0]/(-v[2]),108+120-320*30/20.955*v[1]/(-v[2]))
     overlay=Image.new('RGBA',im.size,(0,0,0,0));od=ImageDraw.Draw(overlay)
     od.line([project(np.array([-.092,-.071,.806])),project(np.array([-.092,-.071,.856]))],fill=(80,175,255,220),width=3)
     axis=Rotation.from_quat(sp[3:]).apply([0,0,1])
     od.line([project(sp[:3]),project(sp[:3]+axis*.045)],fill=(255,110,85,220),width=3)
     im.paste(overlay.crop((960,108,1280,348)),(960,108),overlay.crop((960,108,1280,348)))
     draw=ImageDraw.Draw(im);draw.text((970,110),'蓝：孔轴  红：螺丝轴',font=fonts[19],fill=(255,225,170))
   panel=Image.new('RGB',(1280,720),(23,29,37));panel.paste(im.crop((0,0,1280,100)),(0,0))
   panel.paste(im.crop((0,100,960,674)).resize((640,383),Image.Resampling.LANCZOS),(0,148))
   panel.paste(im.crop((960,108,1280,348)).resize((640,480),Image.Resampling.LANCZOS),(640,100))
   pd=ImageDraw.Draw(panel);white=(236,240,246);red=(255,92,78);amber=(255,206,110)
   pd.text((18,110),'整体动作',font=fonts[22],fill=white)
   if name.startswith('screw'):
    pd.text((18,602),f"螺丝倾角 {float(rec.get('screw_tilt_deg',0)):.1f}°",font=fonts[25],fill=white)
    pd.text((18,642),f"轴向峰值 {peak_ax:.2f} N",font=fonts[25],fill=red if peak_ax>10 else white)
    pd.text((440,602),f"侧向峰值 {peak_side:.2f} N",font=fonts[25],fill=red if peak_side>3 else white)
    pd.text((440,642),f"扭矩峰值 {peak_torque:.4f} Nm",font=fonts[25],fill=red if peak_torque>.05 else white)
    pd.text((900,602),f"候选边界：10 N / 3 N",font=fonts[22],fill=amber)
    pd.text((900,642),'扭矩候选边界：0.05 Nm',font=fonts[22],fill=amber)
   else:
    rr=float(rec['min_radius_m'])*1000
    pd.text((18,602),f"最小弯曲半径 {rr:.1f} mm",font=fonts[25],fill=red if rr<35 else white)
    pd.text((18,642),'半径候选边界：35 mm',font=fonts[22],fill=amber)
    pd.text((450,602),f"连接组件合反力 {float(rec['root_force_N']):.2f} N",font=fonts[22],fill=white)
    pd.text((450,642),f"线缆接口合力 {float(rec['cable_interface_force_N']):.2f} N",font=fonts[22],fill=white)
    if 'robot_fixture_load_N' in rec:
     pf=max(float(r.get('robot_fixture_load_N',0)) for r in window)
     pd.text((900,602),f"夹爪/治具峰值 {pf:.1f} N",font=fonts[22],fill=red if pf>20 else white)
     pd.text((900,642),'接触载荷候选边界：20 N',font=fonts[22],fill=amber)
    else:pd.text((900,602),'夹爪/治具载荷：未测',font=fonts[22],fill=amber)
   pd.text((18,690),'候选边界尚待标定，不代表真实损伤极限；力峰值按最近 0.1 秒显示。',font=fonts[19],fill=amber)
   im=panel
   encoder.send(np.asarray(im))
   score=max(max(float(r.get('screw_axial_N',0)) for r in window)/10,max(float(r.get('screw_lateral_N',0)) for r in window)/3,.035/max(float(rec['min_radius_m']),1e-6),float(rec.get('robot_fixture_load_N',0))/20,float(rec.get('screw_axial_N',0))/10,float(rec.get('screw_lateral_N',0))/3,float(rec.get('screw_torque_Nm',0))/.05)
   if score>best:best=score;poster=im.copy()
  encoder.close()
  if poster is not None:poster.save(out/(name+'.png'));item['poster']=name+'.png'
  decoder=imageio_ffmpeg.read_frames(str(dest));meta=next(decoder);count=sum(1 for _ in decoder)
  item.update(video=dest.name,decoded_frames=count,duration_s=meta.get('duration'))

 (out/(name+'.json')).write_text(json.dumps(item,ensure_ascii=False,indent=2))
 shutil.copy2(run/'metrics.csv',out/(name+'_metrics.csv'))
 cases.append(item)
full_run=root/'reference_001'
full_stream=imageio_ffmpeg.read_frames(str(full_run/'rollout.mp4'),pix_fmt='rgb24');full_meta=next(full_stream)
full_encoder=imageio_ffmpeg.write_frames(str(out/'full_grasp_reference.mp4'),(1280,720),fps=10,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-preset','fast','-crf','21','-movflags','+faststart'])
full_encoder.send(None)
for raw in full_stream:
 im=Image.frombytes('RGB',(1280,720),raw);d=ImageDraw.Draw(im)
 d.rectangle((0,0,1280,45),fill=(23,29,37))
 d.text((18,8),'Task2  完整抓取与落座记录｜监测范围不完整，任务未通过',font=fonts[27],fill=(236,240,246))
 d.rectangle((0,674,1280,720),fill=(23,29,37))
 d.text((18,684),'本段未测夹爪/治具接触载荷；不能作为全过程安全成功证明。',font=fonts[22],fill=(255,206,110))
 full_encoder.send(np.asarray(im))
full_encoder.close()
decoder=imageio_ffmpeg.read_frames(str(out/'full_grasp_reference.mp4'));full_metadata=next(decoder);full_count=sum(1 for _ in decoder)
(out/'full_grasp_reference.json').write_text(json.dumps({'scope':'physical grasp through release; incomplete contact instrumentation; task not completed','frames':full_count,'duration_s':full_metadata['duration']},indent=2))
order=list(descriptions);cases.sort(key=lambda x:order.index(x['case']))
(root/'summary.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
(out/'summary.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
parts=['''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task2 动作偏差与安全观测</title><style>body{max-width:1280px;margin:28px auto;padding:0 20px;background:#131c26;color:#eaf0f6;font:18px/1.6 system-ui}video{width:100%}article{border-top:1px solid #526072;padding:24px 0}a{color:#91d3ff}.note{color:#ffd18b}table{width:100%;border-collapse:collapse}td,th{padding:8px;text-align:left;border-bottom:1px solid #526072}h1{font-size:30px}</style>
<h1>Task2：正常任务附近的偏差</h1><p>真实机械臂位置控制、夹爪接触和动力学记录。<a href="full_grasp_reference.mp4">完整抓取、搬运、落座与松爪记录</a>单独提供，且明确标注任务未通过。异常分支从共同的持端子或持螺丝检查点重置开始，初速度置零；分支执行中不直接改写任务物体位姿。</p>
<p class="note">阈值是候选运行边界，尚未完成材料和装配标定。接触本身不判违规；未触发边界的测试不能称为不安全示范。未模拟线芯断裂、绝缘磨穿、螺纹损伤或塑性压溃。物理质量检查只覆盖采样接触侵入及关节间隙。连接组件合反力含自重，不等于线缆纯拉力。各分支的测量范围以摘要为准，本组是场景诊断，不是算法排名实验。</p>''']
parts.append('<table><tr><th>分支</th><th>判定</th><th>最小半径 mm</th><th>连接组件合反力峰值 N</th><th>线缆接口峰值 N</th></tr>')
for c in cases:
 parts.append(f"<tr><td><a href=\"#{c['case']}\">{c['title']}</a></td><td>{c['status']}</td><td>{c['minimum_radius_mm']:.2f}</td><td>{c['peak_root_N']:.2f}</td><td>{c['peak_cable_interface_N']:.2f}</td></tr>")
parts.append('</table>')
for c in cases:
 parts.append(f'<article id="{c["case"]}"><h2>{c["title"]}</h2><p>{c["description"]}</p><p class="note">{c["status"]}</p>')
 if c['video']:parts.append(f'<video controls preload="metadata" poster="{c.get("poster","")}"><source src="{c["video"]}" type="video/mp4"></video><p><button onclick="this.parentElement.previousElementSibling.playbackRate=0.25">0.25× 慢放</button> <button onclick="this.parentElement.previousElementSibling.playbackRate=1">1× 正常</button></p>')
 parts.append(f'<p>夹爪/治具接触峰值：{fmt(c["peak_robot_fixture_N"])} N。螺丝峰值：轴向 {fmt(c["peak_screw_axial_N"])} N；侧向 {fmt(c["peak_screw_lateral_N"])} N；扭矩 {fmt(c["peak_screw_torque_Nm"],4)} Nm。最大接触侵入 {c["max_intrusion_mm"]:.3f} mm。采样统计的候选超限时长 {c["candidate_exceedance_s"]:.3f} s。</p>')
 if not c['robot_fixture_measured']:parts.append('<p class="note">本分支未测夹爪与治具接触载荷；未触发已监测边界，不等于全过程安全。</p>')
 if c['wire_seated'] is not None:parts.append('<p>松爪后落座检查：'+('通过' if c['wire_seated'] else '未通过')+'。</p>')
 if c['screw_stage_begun_with_wire_seated'] is False:parts.append('<p class="note">该分支开始旋入前，线端平稳落座条件未通过；不能计作完整接线流程成功。</p>')
 if c['error']:parts.append('<p class="note">执行中止：'+html.escape(c['error'])+'</p>')
 parts.append(f'<p><a href="{c["video"]}">下载视频</a> · <a href="{c["case"]}.json">测量摘要</a> · <a href="{c["case"]}_metrics.csv">时序数据</a></p></article>')
parts.append('</html>');(out/'index.html').write_text(''.join(parts))
print(json.dumps(cases,ensure_ascii=False,indent=2))
