"""Save topview MP4 for each pure action on cat_d and person_g L0.

For each (scene × action), reset env, command action for N steps, capture topview
frame per step, write MP4 to /home/jisu/workspace/safety/robotics-safety/tmp/.
"""
import os
import numpy as np
import imageio

os.environ.setdefault('MUJOCO_GL', 'egl')

from envs.robocasa_env import VoxPoserRobocasa
from utils.arguments import get_config

N_STEPS = 60
MAG = 10.0
W, H = 640, 480
OUT_DIR = '/workspace/tmp/action_mp4'    # inside docker → host robotics-safety/tmp/action_mp4

SCENES = [
    ("NavigateKitchenCatBlockingRouteD", 0, 3, 'cat_d'),
    ("NavigateKitchenHumanBlockingRouteG", 0, 3, 'person_g'),
]

CASES = [
    ([MAG, 0, 0],   "vx_pos"),
    ([-MAG, 0, 0],  "vx_neg"),
    ([0, MAG, 0],   "vy_pos"),
    ([0, -MAG, 0],  "vy_neg"),
    ([0, 0, 1.0],   "omega_pos"),
    ([0, 0, -1.0],  "omega_neg"),
]


def render_topview(env):
    return env.env.sim.render(camera_name='topview', width=W, height=H)[::-1]


def trace(env, action, fps=20):
    env.env.reset()
    frames = [render_topview(env)]
    for _ in range(N_STEPS):
        env.apply_navigation_action(np.array(action))
        frames.append(render_topview(env))
    return frames


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for task, layout, style, label in SCENES:
        print(f"\n=== {label} ===")
        config = get_config(config_path='configs/robocasa_config.yaml', task_type='navigation')
        task_cfg = config['task']
        task_cfg['layout_ids'] = layout
        task_cfg['style_ids'] = style
        env = VoxPoserRobocasa(task_name=task, task_config=task_cfg)
        for action, name in CASES:
            print(f"  → {name}")
            frames = trace(env, action)
            out = os.path.join(OUT_DIR, f"{label}_{name}.mp4")
            imageio.mimwrite(out, frames, fps=20, codec='libx264', quality=8)
            print(f"    saved {out}  ({len(frames)} frames)")
    print("\ndone.")


if __name__ == "__main__":
    main()
