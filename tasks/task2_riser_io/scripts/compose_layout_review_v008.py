from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/layout_review_v008')
cfg=json.loads((O/'cameras.json').read_text());layout=json.loads((O/'layout.json').read_text())
fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font=lambda n:ImageFont.truetype(fontpath,n)
im=Image.new('RGB',(1800,1510),(242,245,248));d=ImageDraw.Draw(im)
dark=(30,47,61);muted=(78,93,106)
d.text((28,18),'Task2 单线安全接线｜机箱边缘与余量管理',font=font(38),fill=dark)
d.text((30,74),'Isaac Sim 静态渲染 · 几何与布局审核 · 物理参数、安全阈值及操作验证待确认后进行',font=font(23),fill=muted)
main=Image.open(O/'overview.png').convert('RGB').resize((1152,864),Image.Resampling.LANCZOS)
im.paste(main,(24,132))
d.rectangle((24,110,1176,148),fill=(34,52,67));d.text((38,114),'工作区布局：取端子 → 搬运落座 → 取螺丝 → 紧固',font=font(23),fill='white')
for name,y,title in [('connection',156,'近端：受保护连接 + 应力释放护套'),('target',650,'目标端：机箱安装板 + 开放护线坡面')]:
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
 ('6',[-.28,-.21,.780],(-60,5),(93,90,125)),
 ('7',[-.092,.085,.787],(-70,8),(25,117,153)),
 ('8',[-.020,.044,.800],(60,-20),(179,63,49))]
for label,p,offset,color in markers:
 a=project(p);b=a+offset
 d.line([tuple(a),tuple(b)],fill=color,width=4);d.ellipse((a[0]-4,a[1]-4,a[0]+4,a[1]+4),fill=color)
 d.ellipse((b[0]-20,b[1]-20,b[0]+20,b[1]+20),fill=color,outline='white',width=3)
 bb=d.textbbox((0,0),label,font=font(24));d.text((b[0]-(bb[2]-bb[0])/2,b[1]-18),label,font=font(24),fill='white')
d.text((34,1012),'同样的落座结果，可能留下不同的余量分布、接触历史和残余载荷。',font=font(24),fill=dark)
entries=[
 ('1  固定连接端','测量连接接口载荷与根部弯矩',(179,63,49)),
 ('2  应力释放护套','支撑出口；弯曲可能转移至护套末端',(25,117,153)),
 ('3  可见服务余量','接触阻滞时，总余量不等于可用余量',(180,116,28)),
 ('4  端子取料托座','保持圆片几何，降低抓取难度',(25,117,153)),
 ('5  单工位紧固座','监测偏载、过紧与紧固带动端子转动',(39,125,77)),
 ('6  单颗螺丝托座','直接取料；不增加换夹具步骤',(93,90,125)),
 ('7  开放护线坡面','跨越台阶并支撑线缆，可滑移或抬离',(25,117,153)),
 ('8  机箱安装板','承载接地端子；边缘加载与擦动可测',(179,63,49))]
for k,(title,desc,color) in enumerate(entries):
 x=24+(k%2)*888;y=1108+(k//2)*84
 d.rounded_rectangle((x,y,x+864,y+73),radius=8,fill='white')
 d.rectangle((x,y+6,x+5,y+67),fill=color)
 d.text((x+18,y+5),title,font=font(24),fill=color)
 d.text((x+18,y+39),desc,font=font(21),fill=dark)
d.text((28,1458),'几何审核版本：边缘摩擦、护套变形、载荷测量和安全可行解尚待物理验证；不代表 NASA 合规认证。',font=font(21),fill=muted)
im.save(O/'layout_review.png')
print(O/'layout_review.png')
