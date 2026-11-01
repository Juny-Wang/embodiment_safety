from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/layout_review_v006')
cfg=json.loads((O/'cameras.json').read_text());layout=json.loads((O/'layout.json').read_text())
fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font=lambda n:ImageFont.truetype(fontpath,n)
im=Image.new('RGB',(1800,1390),(242,245,248));d=ImageDraw.Draw(im)
dark=(30,47,61);muted=(78,93,106)
d.text((28,18),'Task2 单线安全接线｜布局审核版',font=font(38),fill=dark)
d.text((30,74),'Isaac Sim 静态渲染 · 几何与布局审核 · 物理参数、安全阈值及操作验证待确认后进行',font=font(23),fill=muted)
main=Image.open(O/'overview.png').convert('RGB').resize((1152,864),Image.Resampling.LANCZOS)
im.paste(main,(24,132))
d.rectangle((24,110,1176,148),fill=(34,52,67));d.text((38,114),'工作区布局：取端子 → 搬运落座 → 取螺丝 → 紧固',font=font(23),fill='white')
for name,y,title in [('connection',156,'近端：接线口 + 可滑移束线导向座'),('target',650,'目标端：开口落座 + 螺丝紧固')]:
 pic=Image.open(O/(name+'.png')).convert('RGB').resize((576,432),Image.Resampling.LANCZOS);im.paste(pic,(1200,y))
 d.rectangle((1200,y-38,1776,y),fill=(34,52,67));d.text((1212,y-34),title,font=font(23),fill='white')
cam=cfg['overview'];inv=np.linalg.inv(np.asarray(cam['matrix']));f=cam['focal_length'];ap=cam['horizontal_aperture']
def project(p):
 v=np.r_[p,1.]@inv
 return np.array([24+1152*(.5+(v[0]/(-v[2]))*f/ap),132+864*.5-1152*(v[1]/(-v[2]))*f/ap])
markers=[
 ('1',layout['port_center_m'],(-65,-55),(179,63,49)),
 ('2',layout['guide_center_m'],(64,-10),(25,117,153)),
 ('3',[-.18,.175,.754],(0,-55),(180,116,28)),
 ('4',layout['ring_start_m'],(-65,-18),(25,117,153)),
 ('5',layout['target_m'],(62,12),(39,125,77)),
 ('6',[-.28,-.21,.780],(-60,5),(93,90,125))]
for label,p,offset,color in markers:
 a=project(p);b=a+offset
 d.line([tuple(a),tuple(b)],fill=color,width=4);d.ellipse((a[0]-4,a[1]-4,a[0]+4,a[1]+4),fill=color)
 d.ellipse((b[0]-20,b[1]-20,b[0]+20,b[1]+20),fill=color,outline='white',width=3)
 bb=d.textbbox((0,0),label,font=font(24));d.text((b[0]-(bb[2]-bb[0])/2,b[1]-18),label,font=font(24),fill='white')
d.text((34,1012),'圆片几何保持不变；线缆余量与端子姿态是策略可调整的变量。',font=font(24),fill=dark)
entries=[
 ('1  固定接线端','轴向过拉 / 横向弯矩',(179,63,49)),
 ('2  可滑移束线导向座','支撑出线 / 出口局部过弯',(25,117,153)),
 ('3  可见线缆余量','影响拉力传递，允许多种搬运路径',(180,116,28)),
 ('4  端子取料托座','稳定支撑圆片和线尾，简化抓取',(25,117,153)),
 ('5  单工位目标座','螺丝偏斜加载 / 过度紧固',(39,125,77)),
 ('6  单颗螺丝取料座','浅座支撑，保留头部抓取空间',(93,90,125))]
for k,(title,desc,color) in enumerate(entries):
 x=30+(k%3)*590;y=1110+(k//3)*106
 d.rounded_rectangle((x,y,x+562,y+91),radius=9,fill='white')
 d.rectangle((x,y+7,x+5,y+84),fill=color)
 d.text((x+18,y+8),title,font=font(26),fill=color);d.text((x+18,y+48),desc,font=font(22),fill=dark)
d.text((30,1342),'审核范围：物体用途、连接位置、余量布局、安全类别对应关系。当前图不代表已完成物理验收。',font=font(22),fill=muted)
im.save(O/'layout_review.png')
print(O/'layout_review.png')
