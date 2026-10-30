"""Pure metrics, no simulator imports. Units: m, N; thresholds are demonstration only."""
import numpy as np
def curvature(points):
 p=np.asarray(points,float)
 if len(p)<3 or not np.isfinite(p).all():return {'min_bend_radius_m':None,'radii_m':[]}
 a=np.linalg.norm(p[1:-1]-p[:-2],axis=1);b=np.linalg.norm(p[2:]-p[1:-1],axis=1);c=np.linalg.norm(p[2:]-p[:-2],axis=1)
 area2=np.linalg.norm(np.cross(p[1:-1]-p[:-2],p[2:]-p[:-2]),axis=1)
 valid=(a>1e-8)&(b>1e-8)&(c>1e-8)
 if not valid.all():return {'min_bend_radius_m':None,'radii_m':[]}
 radii=np.divide(a*b*c,2*area2,out=np.full_like(a,float('inf')),where=area2>1e-12)
 return {'min_bend_radius_m':float(np.min(radii)) if np.isfinite(radii).any() else None,'straight':not np.isfinite(radii).any(),'radii_m':[float(x) if np.isfinite(x) else None for x in radii]}
def summarize(points,anchor_wrench,limit_N=1.,min_radius=.035):
 p=np.asarray(points,float);c=curvature(p)
 length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
 chord=float(np.linalg.norm(p[-1]-p[0]))
 available=anchor_wrench is not None and np.isfinite(anchor_wrench).all()
 force=float(np.linalg.norm(anchor_wrench[:3])) if available else None
 moment=float(np.linalg.norm(anchor_wrench[3:])) if available else None
 ratios=[];reasons=[]
 if force is not None:
  ratios.append(force/limit_N)
  if force>=limit_N:reasons.append('CONNECTOR_LOAD_DEMO_LIMIT')
 if c['min_bend_radius_m'] is not None:
  ratios.append(min_radius/c['min_bend_radius_m'])
  if c['min_bend_radius_m']<min_radius:reasons.append('BEND_RADIUS_DEMO_LIMIT')
 light='GRAY' if not available else ('RED' if max(ratios or [0])>=1 else 'YELLOW' if max(ratios or [0])>=.7 else 'GREEN')
 return {**c,'sampled_centerline_length_m':length,'chord_m':chord,'geometric_slack_m':max(0.,length-chord),'connector_force_N':force,'connector_moment_Nm':moment,'state':light,'reasons':reasons,'stress_Pa':None,'damage':None,'threshold_source':'DEMONSTRATION_ONLY','mode':'warning_only'}
