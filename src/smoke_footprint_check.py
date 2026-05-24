"""_get_fixture_floor_footprint 재현 — fridge_g1을 scene_collision이 덮는지 확인.
run: cd src && MUJOCO_GL=egl python3 smoke_footprint_check.py
"""
import os
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
from envs.robocasa_env import VoxPoserRobocasa
from utils.arguments import get_config

config = get_config(config_path='configs/robocasa_config.yaml', task_type='navigation')
tc = config['task']; tc['layout_ids'] = 0; tc['style_ids'] = 3
env = VoxPoserRobocasa(task_name='NavigateKitchenWineBlockingRouteA', task_config=tc)
env.env.reset()
sim = env.env.sim; model = sim.model

wmin = np.asarray(env.workspace_bounds_min[:2], float)
wmax = np.asarray(env.workspace_bounds_max[:2], float)
H, W = 122, 112   # dump avoidance_map shape
sx_m = (wmax[0] - wmin[0]) / W
sy_m = (wmax[1] - wmin[1]) / H
print(f"workspace x{wmin[0]:.2f}~{wmax[0]:.2f} y{wmin[1]:.2f}~{wmax[1]:.2f}  cell {sx_m:.3f}x{sy_m:.3f}m")

exclude_patterns = ('robot0','mobilebase','gripper0','panda','eef_target','_target','world')
exclude_body_ids = set()
for i in range(model.nbody):
    n = (model.body_id2name(i) or '').lower()
    if any(p in n for p in exclude_patterns):
        exclude_body_ids.add(i)

mask = np.zeros((H, W), dtype=bool)
fridge_geoms = []
for gid in range(model.ngeom):
    gname = (model.geom_id2name(gid) or '').lower()
    if 'floor' in gname:
        continue
    bid = int(model.geom_bodyid[gid])
    if bid in exclude_body_ids:
        continue
    p = sim.data.geom_xpos[gid]
    if p[2] < -0.05 or p[2] > 1.8:
        skip_z = True
    else:
        skip_z = False
    gs = model.geom_size[gid]
    hx, hy = float(gs[0]), float(gs[1])
    is_fridge = 'fridge' in gname
    if is_fridge:
        fridge_geoms.append((gname, p[:3].copy(), gs.copy(), skip_z))
    if skip_z or hx <= 0 or hy <= 0:
        continue
    mat = sim.data.geom_xmat[gid].reshape(3, 3)
    half_x = abs(mat[0,0])*hx + abs(mat[0,1])*hy
    half_y = abs(mat[1,0])*hx + abs(mat[1,1])*hy
    x0,x1 = p[0]-half_x, p[0]+half_x
    y0,y1 = p[1]-half_y, p[1]+half_y
    x0=max(x0,wmin[0]); x1=min(x1,wmax[0]); y0=max(y0,wmin[1]); y1=min(y1,wmax[1])
    if x1<=x0 or y1<=y0: continue
    c0=int(np.floor((x0-wmin[0])/sx_m)); c1=int(np.ceil((x1-wmin[0])/sx_m))
    r0=int(np.floor((y0-wmin[1])/sy_m)); r1=int(np.ceil((y1-wmin[1])/sy_m))
    r0=max(0,r0); r1=min(H,r1); c0=max(0,c0); c1=min(W,c1)
    if r1<=r0 or c1<=c0: continue
    mask[r0:r1, c0:c1] = True

print(f"\nfootprint mask obstacle 셀: {int(mask.sum())}/{mask.size}")
print(f"\n=== fridge geom들 ===")
for gname, p, gs, skip_z in fridge_geoms:
    print(f"  {gname:32} pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) size={np.round(gs,2)}"
          f"  {'★z범위밖→스킵' if skip_z else ''}{' ★hx/hy=0→스킵' if (gs[0]<=0 or gs[1]<=0) else ''}")

def px(x,y):
    return int((y-wmin[1])/sy_m), int((x-wmin[0])/sx_m)
for nm,(x,y) in [('fridge g1 중심',(5.00,-0.69)),('wedge',(5.15,-0.97)),('robot start',(5.01,-1.03))]:
    r,c = px(x,y)
    v = mask[r,c] if 0<=r<H and 0<=c<W else None
    print(f"  {nm:14} ({x:.2f},{y:.2f}) px(r{r},c{c}) footprint={v}")
