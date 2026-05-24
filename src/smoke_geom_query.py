"""L0 Wine RouteA env에서 로봇 wedge 좌표 주변 geom 조회.
로봇이 박힌 (5.15,-0.97)에 실제 정적 geom(카운터/벽)이 있는지 정의적 확인.
run: cd src && MUJOCO_GL=egl python3 smoke_geom_query.py
"""
import os
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
from envs.robocasa_env import VoxPoserRobocasa
from utils.arguments import get_config

WEDGE = np.array([5.15, -0.97])   # L0 Wine RouteA 로봇 wedge 위치
RADIUS = 0.7                       # 이 반경 내 geom 조회

config = get_config(config_path='configs/robocasa_config.yaml', task_type='navigation')
tc = config['task']
tc['layout_ids'] = 0
tc['style_ids'] = 3
env = VoxPoserRobocasa(task_name='NavigateKitchenWineBlockingRouteA', task_config=tc)
env.env.reset()
sim = env.env.sim
model = sim.model

obs = env.env._get_observations()
print(f"robot start pos = {obs['robot0_base_pos'][:2]}")
print(f"wedge 조회점 = {WEDGE}, 반경 {RADIUS}m\n")

hits = []
for gid in range(model.ngeom):
    p = sim.data.geom_xpos[gid]
    if p[2] < -0.05 or p[2] > 1.8:
        continue
    d = float(np.linalg.norm(p[:2] - WEDGE))
    if d <= RADIUS:
        gname = model.geom_id2name(gid) or f'geom{gid}'
        bid = int(model.geom_bodyid[gid])
        bname = model.body_id2name(bid) or f'body{bid}'
        sz = model.geom_size[gid]
        hits.append((d, gname, bname, p[:2].copy(), sz.copy(), p[2]))

hits.sort()
print(f"=== wedge {RADIUS}m 내 geom {len(hits)}개 ===")
for d, gname, bname, p, sz, z in hits:
    print(f"  d={d:.2f}m  geom={gname:28} body={bname:24} "
          f"xy=({p[0]:.2f},{p[1]:.2f}) z={z:.2f} size={np.round(sz,2)}")
if not hits:
    print("  (없음 — wedge 지점에 정적 geom 없음 → 물리적 벽 아님)")
