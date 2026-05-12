"""Action scaling sweep: MAG ∈ [0.3, 1.0, 3.0, 10.0, 30.0, 100.0] for pure axes.

Checks whether action is interpreted linearly (m/s-like) or saturates.
Linear scaling → can fix attenuation with a simple multiplier.
Saturation point → that's the controller's true max output.

Usage: cd src && python3 smoke_action_scaling.py
"""
import os
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')

from envs.robocasa_env import VoxPoserRobocasa
from utils.arguments import get_config
from transforms3d.euler import quat2euler

TASK = "NavigateKitchenCatBlockingRouteD"
LAYOUT = 0
STYLE = 3
N_STEPS = 50


def read(env):
    obs = env.env._get_observations()
    q = obs['robot0_base_quat']
    yaw = quat2euler([q[3], q[0], q[1], q[2]])[2]
    pos = obs['robot0_base_pos'][:2].copy()
    return float(yaw), pos


def run_case(env, action, label):
    env.env.reset()
    yaw0, pos0 = read(env)
    for _ in range(N_STEPS):
        env.apply_navigation_action(np.array(action))
    yawf, posf = read(env)
    dpos_world = posf - pos0
    dpos_fwd = dpos_world[0]*np.cos(yaw0) + dpos_world[1]*np.sin(yaw0)
    dpos_lat = -dpos_world[0]*np.sin(yaw0) + dpos_world[1]*np.cos(yaw0)
    dyaw = (yawf - yaw0 + np.pi) % (2*np.pi) - np.pi
    print(f"  {label:<26}  fwd={dpos_fwd*1000:+7.1f}mm lat={dpos_lat*1000:+7.1f}mm  Δyaw={np.degrees(dyaw):+6.1f}°")


def main():
    config = get_config(config_path='configs/robocasa_config.yaml', task_type='navigation')
    task_cfg = config['task']
    task_cfg['layout_ids'] = LAYOUT
    task_cfg['style_ids'] = STYLE
    env = VoxPoserRobocasa(task_name=TASK, task_config=task_cfg)
    env.env.reset()
    yaw0, pos0 = read(env)
    print(f"\n[scaling] start=({pos0[0]:.3f},{pos0[1]:.3f}) yaw0={np.degrees(yaw0):.1f}°"
          f" N={N_STEPS} ({N_STEPS/20:.1f}s sim)\n")

    MAGS = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    print(f"--- pure +v_x ---")
    for m in MAGS:
        run_case(env, [m, 0, 0], f"v_x={m}")
    print(f"\n--- pure +v_y ---")
    for m in MAGS:
        run_case(env, [0, m, 0], f"v_y={m}")
    print(f"\n--- pure +omega ---")
    for m in MAGS:
        run_case(env, [0, 0, m], f"omega={m}")


if __name__ == "__main__":
    main()
