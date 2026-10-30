"""Reserved navigation keys; shift selects world-axis rotation.

Gripper: [ / O open, ] / P close (letter aliases for remote-desktop layouts).
"""
import math
TRANSLATION={"UP":(0,1),"DOWN":(0,-1),"LEFT":(1,1),"RIGHT":(1,-1),"PAGE_UP":(2,1),"PAGE_DOWN":(2,-1)}
OPEN_KEYS={"LEFT_BRACKET","O"}
CLOSE_KEYS={"RIGHT_BRACKET","P"}
KEYS=set(TRANSLATION)|OPEN_KEYS|CLOSE_KEYS

def vectors(held):
 v=[0.,0.,0.]
 for key in held:
  if key in TRANSLATION:
   axis,sign=TRANSLATION[key];v[axis]+=sign
 norm=math.sqrt(sum(x*x for x in v))
 v=[x/max(1.,norm) for x in v]
 return [[0.,0.,0.],v] if held & {"LEFT_SHIFT","RIGHT_SHIFT"} else [v,[0.,0.,0.]]

def gripper_sign(held):
 """+1 open, -1 close, 0 none/cancel. Letters O/P alias brackets."""
 open_h=bool(held & OPEN_KEYS); close_h=bool(held & CLOSE_KEYS)
 if open_h==close_h: return 0
 return 1 if open_h else -1
