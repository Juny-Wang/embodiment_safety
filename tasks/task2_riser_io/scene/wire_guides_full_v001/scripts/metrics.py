"""Pure monitoring logic. No force -> stress conversion, no robot commands."""
import numpy as np
def evaluate(points,wrench,guide_normal_N,cfg):
 p=np.asarray(points,dtype=float);w=np.asarray(wrench,dtype=float) if wrench is not None else None
 if p.shape!=(24,3) or not np.isfinite(p).all() or (w is not None and (w.shape!=(6,) or not np.isfinite(w).all())):raise ValueError('invalid/nonfinite physics data')
 a=np.linalg.norm(p[1:-1]-p[:-2],axis=1);b=np.linalg.norm(p[2:]-p[1:-1],axis=1);c=np.linalg.norm(p[2:]-p[:-2],axis=1)
 if min(a.min(),b.min(),c.min())<1e-8:raise ValueError('degenerate cable centerline')
 cross=np.linalg.norm(np.cross(p[1:-1]-p[:-2],p[2:]-p[:-2]),axis=1)
 rr=np.divide(a*b*c,2*cross,out=np.full_like(a,np.inf),where=cross>1e-12)
 rmin=float(rr.min()) if np.isfinite(rr).any() else None
 force=float(np.linalg.norm(w[:3])) if w is not None else None;moment=float(np.linalg.norm(w[3:])) if w is not None else None
 if (force is not None and force>1000) or np.max(np.linalg.norm(p-p[0],axis=1))>1.5:raise ValueError('diagnostic plausibility limit exceeded; not material damage')
 ratios=[force/cfg['joint_reaction_N']] if force is not None else []
 reasons=[]
 if force is not None and force>=cfg['joint_reaction_N']:reasons.append('NEAR_END_JOINT_LOAD_DEMO_LIMIT')
 if rmin is not None:
  ratios.append(cfg['min_bend_radius_m']/rmin)
  if rmin<cfg['min_bend_radius_m']:reasons.append('BEND_RADIUS_DEMO_LIMIT')
 if guide_normal_N is not None:
  if not np.isfinite(guide_normal_N) or guide_normal_N<0:raise ValueError('invalid guide force')
  ratios.append(guide_normal_N/cfg['guide_normal_N'])
  if guide_normal_N>=cfg['guide_normal_N']:reasons.append('GUIDE_CONTACT_DEMO_LIMIT')
 color='RED' if max(ratios or [0])>=1 else 'YELLOW' if max(ratios)>=cfg['warning_ratio'] else 'GREEN'
 return {'state':color,'reasons':reasons,'near_end_joint_force_N':force,'near_end_joint_moment_Nm':moment,'guide_normal_N':guide_normal_N,'min_bend_radius_m':rmin,'radii_m':[float(v) if np.isfinite(v) else None for v in rr],'stress_Pa':None,'damage':None,'true_connector_root_force_N':None,'threshold_source':cfg['source'],'warning_only':True,'internal_load_available':w is not None,'assessment_scope':'contact_and_bend_only' if w is None else 'joint_contact_and_bend'}
