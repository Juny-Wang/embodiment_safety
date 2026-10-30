from pathlib import Path
import json,math
import numpy as np
from pxr import Usd,UsdGeom,Gf
from PIL import Image,ImageDraw,ImageFont
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_guides_full_v001')
cfg=json.loads((P/'config.json').read_text());S=P.parent
ns={};exec((P/'scripts/metrics.py').read_text(),ns);evaluate=ns['evaluate']
pts=np.c_[np.linspace(0,.46,24),np.zeros((24,2))];tests=[]
for f,state in [(0,'GREEN'),(.7,'YELLOW'),(1.,'RED')]:
 m=evaluate(pts,np.array([f,0,0,0,0,0]),0,cfg['thresholds']);assert m['state']==state;tests.append({'force_N':f,'state':state})
assert evaluate(pts,np.zeros(6),1.01,cfg['thresholds'])['state']=='RED'
try:evaluate(pts,np.full(6,np.nan),0,cfg['thresholds']);raise AssertionError('NaN accepted')
except ValueError:pass
(P/'reports/logic_checks.json').write_text(json.dumps({'checks':tests,'guide_force_limit_pass':True,'nonfinite_rejected':True,'warning_only':True},indent=2))
stage=Usd.Stage.Open(cfg['scene']);xc=UsdGeom.XformCache();bc=UsdGeom.BBoxCache(0,['default','render','proxy'])
im=Image.new('RGB',(1500,1050),'#eef3f6');d=ImageDraw.Draw(im);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
def xy(p):return (int(70+(p[0]+.6)*1100),int(100+(.42-p[1])*1000))
d.text((45,20),'TASK2 / COMPLETE SCENE COPY + PRINTED GUIDES',font=font,fill='#173349')
d.rectangle([xy([-.6,.407]),xy([.6,-.393])],fill='#dde2e4',outline='#647685',width=3)
colors={'/PCB_holder':'#8097a0','/PCB_4_Terminal':'#467a6e','/Electronics_140mm_19mm_cage':'#8799a9','/TE_L26620011_4_connector_v2':'#a4aaaf','/ScrewTray':'#537ea2'}
for root,color in colors.items():
 for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
  if not p.IsA(UsdGeom.Mesh):continue
  u=UsdGeom.Mesh(p);m=xc.GetLocalToWorldTransform(p);points=[xy(m.Transform(Gf.Vec3d(v))) for v in u.GetPointsAttr().Get()]
  ids=list(u.GetFaceVertexIndicesAttr().Get());offset=0
  for n in u.GetFaceVertexCountsAttr().Get():d.polygon([points[k] for k in ids[offset:offset+n]],fill=color);offset+=n
for w,color in [(1,'#c7673a'),(2,'#478bb3'),(3,'#4a9466'),(4,'#9c70ae')]:
 centers=[xc.GetLocalToWorldTransform(stage.GetPrimAtPath(f'/wire{w}/Segment_{i:02d}')).ExtractTranslation() for i in range(24)]
 d.line([xy(p) for p in centers],fill=color,width=8)
 end=xy(centers[-1]);d.ellipse([end[0]-10,end[1]-10,end[0]+10,end[1]+10],outline=color,width=3);d.text((end[0]-20,end[1]+18),f'wire{w}',font=small,fill=color)
for i,path in enumerate(cfg['guide_paths'],1):
 bound=bc.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange();lo=bound.GetMin();hi=bound.GetMax();center=(lo+hi)*.5
 d.rectangle([xy([lo[0],hi[1]]),xy([hi[0],lo[1]])],fill='#e3b663',outline='#895514',width=2)
 c=xy(center);d.ellipse([c[0]-20,c[1]-18,c[0]+20,c[1]+18],fill='#dc8f28',outline='#895514',width=3)
 d.text((c[0]-55,c[1]+34),f'Guide {i} D36',font=small,fill='#704511')
robot=xy([-.43,0]);d.ellipse([robot[0]-50,robot[1]-50,robot[0]+50,robot[1]+50],fill='#c8ced5',outline='#647685',width=2);d.text((robot[0]-65,robot[1]+55),'Franka base',font=small,fill='#18364c')
for text,pos in [('Connector / 4 anchors',[-.05,.39]),('Cage',[-.04,.21]),('Ring seats + nuts',[-.11,-.12]),('Screw tray',[.17,-.28])]:d.text(xy(pos),text,font=small,fill='#173349')
d.text((45,955),'Top view from composed USD. Robot arm omitted from this layout diagram.',font=small,fill='#173349')
d.text((45,987),'DEMO limits: guide 1 N | bend radius 35 mm. Internal load / material stress unavailable.',font=small,fill='#97462b');im.save(P/'assembly_layout.png')
readme="""# 完整 Task2 导向柱副本 v001

打开同级 riser_io_wire_guides_v001.usda。这是完整主场景副本，保留 Franka、四根 wire、环端子、端子板及螺丝托盘。原 riser_io_preview.usda 未修改。

在 Isaac Script Editor 执行下列一行，再 Play：
    exec(compile(open('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_guides_full_v001/scripts/launch_monitor.py').read(), 'task2_wire_monitor', 'exec'))

键盘操控仍使用原 scene/launch_panel.py。监测器不发机器人命令、不暂停仿真，颜色只写 session layer。
暂停或读数不可用时显示灰色/UNAVAILABLE，不以零替代；日志在 runs/gui_*/measurements.jsonl。
原操控面板若同时对线缆着色，可能覆盖场景颜色；以本监测面板数值为准。

## 信号与演示阈值
- 内部关节反力、轴向张力、材料应力：本版均不可用，面板显示 N/A。
- 导柱：实际接触点法向力，按 physics dt 换算；不代表全部摩擦力。
- 弯曲：实时段中心三点外接圆半径，属于离散近似，不是材料纤维应变。
- 演示阈值：导柱力 1 N、弯曲半径下限 35 mm；黄灯 70%，红灯 100%。配置中的 joint 1 N 预留阈值未启用。
- 颜色由实测数值触发；没有材料损伤判据，不生成断裂，也不把绿色当强度认证。
- 暂不自动区分允许导向接触与禁区损伤。

## 副本的修正
原 wire 的运动学首节被挂了 articulation API，PhysX 忽略它。本副本移除了这项无效 API，
保留原运动学首节和普通关节链；只规范化四个末端固定关节的非单位四元数，增加接触上报和明确材质。
动态首节/固定锚点、关节重对齐及米制展开的实验均未通过，已撤回，详见 reports/rejected_articulation_metres.json。
不要运行 repair_sensors.py 或 bake_units.py 来覆盖交付场景；它们仅保留为失败诊断证据。
本版不能测量线内部载荷，不能判断真实材料应力是否达到极限。
Franka 初始属性没有修改。完整 19 步、绕柱路线和材料应力未因此验收。
导柱网格与 STL 一致，使用静态三角形碰撞，材质摩擦 0.4/0.3、恢复系数 0，均未实物标定。

## 打印和装配
STL 单位 mm，切片比例 100%；尺寸见 reports/print_geometry_final.json，数量见 BOM.json。
任务零件是机械教学替身，没有真实电气功能。Franka 与桌面是外部实验平台，不属于本套打印件。
刚性件建议 PETG，柔性线建议 TPU；具体工艺、方向、间隙需要试印。
1. 导柱 2，长螺丝 4，螺母 4。桌面厚度按场景 50 mm；按 static_audit.json 的 XY 坐标开四个 D11.4 通孔。
   用 D10/P2、杆长 66 mm 的打印螺丝固定，螺母在桌下。实际桌面不能钻孔时须重新设计安装工装。
2. 端子板、底座、托盘各 1；环端子、短螺丝及配套螺母各 4；笼子和连接器各 1。
3. 四条 TPU U 形线各 1，外径 7 mm，展开长度约 455–459 mm；U 形只是平放打印布局，使用前柔性展开。
   原环端子为 7 mm 线缆卡装替身。TPU 的拉伸/弯曲响应、接口夹紧与拔脱力尚未标定，不能当作力学等价件。
4. 先试印粗牙配合件。封闭网格不证明螺纹强度、薄壁可打印或装配可靠。

连接器的三个共线 T 接缝已通过现有顶点重新三角化封闭，不移动表面。
reference_only 保存未修复网格与过长直线线缆，不属于建议打印件。
笼子和连接器保留源外形，薄壁、打印支撑及接口夹持仍需切片和试装。
本包保留全打印搭建目标，不能标为已通过实物验收。

## 验证
reports/full_smoke.json 是交付副本的原生 PhysX 接触/几何检查，不是内部载荷测量验收；早期失败另行保留。
logic_checks.json 仅验证阈值逻辑。assembly_layout.png 是实际 USD 世界坐标布局图，不是成功动作截图。
源场景哈希、Franka 属性对比、补丁及文件哈希随包提供。主场景未写回，未训练。
"""
(P/'README.md').write_text(readme)
counts={'guide_column_D36_H35.stl':2,'guide_mount_screw_D10_P2_shaft66.stl':4,'nut_custom_D10_P2_clearance_030.stl':8,'ring_terminal_surrogate.stl':4,'screw_custom_D10_P2.stl':4}
bom=[{'file':'stl/'+f.name,'quantity':counts.get(f.name,1),'material':'TPU candidate' if 'TPU' in f.name else 'PETG candidate','units':'mm','physical_print_and_assembly_verified':False} for f in sorted((P/'stl').glob('*.stl'))]
(P/'BOM.json').write_text(json.dumps(bom,indent=2));print('DOCUMENTS_AND_LOGIC_READY')
