"""Topview에 planner waypoints만 깔끔하게 overlay.

각 wp의 world 좌표 → topview pixel 변환하여 번호 + 화살표 표시.
이전 run의 voxposer_dump.npz 로딩.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault('MUJOCO_GL', 'egl')

from envs.robocasa_env import VoxPoserRobocasa
from utils.arguments import get_config
from transforms3d.euler import quat2euler

OUT_DIR = '/workspace/tmp/wp_overlay'
W, H = 1280, 960

CASES = [
    ("NavigateKitchenCatBlockingRouteD", 0, 3, 'cat_d',
     '/workspace/policy/Voxposer/outputs/smoke_optB_114242/cat_d/layout0/NavigateKitchenCatBlockingRouteD/voxposer_dump.npz'),
    ("NavigateKitchenHumanBlockingRouteG", 0, 3, 'person_g',
     '/workspace/policy/Voxposer/outputs/smoke_optB_114242/person_g/layout0/NavigateKitchenHumanBlockingRouteG/voxposer_dump.npz'),
]


def w2p(xy, cam_pos, cam_mat, fovy, W, H, fz):
    fy = (H/2.0) / np.tan(np.radians(fovy/2.0))
    fx = fy
    w = np.array([xy[0], xy[1], fz])
    rel = w - cam_pos
    cf = cam_mat.T @ rel
    depth = -cf[2]
    u = (W/2.0) + fx * (cf[0]/depth)
    v = (H/2.0) - fy * (cf[1]/depth)
    return float(u), H - float(v)


def render(task, layout, style, label, dump_path):
    config = get_config(config_path='configs/robocasa_config.yaml', task_type='navigation')
    task_cfg = config['task']
    task_cfg['layout_ids'] = layout
    task_cfg['style_ids'] = style
    env = VoxPoserRobocasa(task_name=task, task_config=task_cfg)
    env.env.reset()
    sim = env.env.sim
    obs = env.env._get_observations()
    q = obs['robot0_base_quat']
    yaw = quat2euler([q[3], q[0], q[1], q[2]])[2]
    robot_xy = obs['robot0_base_pos'][:2]
    goal_xy = np.array(env.env.target_pos[:2])

    cid = sim.model.camera_name2id('topview')
    cam_pos = sim.data.cam_xpos[cid].copy()
    cam_mat = sim.data.cam_xmat[cid].reshape(3, 3).copy()
    fovy = float(sim.model.cam_fovy[cid])
    fz = 0.0
    try:
        fids = getattr(env, 'floor_mask_ids', [])
        if fids:
            fz = float(sim.data.geom_xpos[fids[0], 2])
    except Exception:
        pass

    img = sim.render(camera_name='topview', width=W, height=H)[::-1]

    # Load waypoints
    d = np.load(dump_path, allow_pickle=True)
    iters = d['iters']
    it0 = iters[0]
    traj_world = it0['traj_world']
    start_pos = it0.get('start_pos', None)

    print(f"\n=== {label} ===")
    print(f"  robot=({robot_xy[0]:.3f},{robot_xy[1]:.3f}) yaw={np.degrees(yaw):.1f}°")
    print(f"  goal =({goal_xy[0]:.3f},{goal_xy[1]:.3f})")
    print(f"  start_pos in dump: {start_pos}")
    print(f"  traj_world ({len(traj_world)} wps):")
    for i, wp in enumerate(traj_world):
        wp_xy = np.asarray(wp)   # dump's traj_world shape (N, 2): row is xy
        # body-frame command if robot at robot_xy facing yaw, target=wp_xy
        dx = wp_xy[0] - robot_xy[0]
        dy = wp_xy[1] - robot_xy[1]
        vxb = dx*np.cos(yaw) + dy*np.sin(yaw)
        vyb = -dx*np.sin(yaw) + dy*np.cos(yaw)
        face = float(np.arctan2(dy, dx))
        d_yaw_face = (face - yaw + np.pi) % (2*np.pi) - np.pi
        print(f"    wp[{i:2d}] world=({wp_xy[0]:+.3f},{wp_xy[1]:+.3f})  "
              f"Δworld=({dx:+.3f},{dy:+.3f})  body=(fwd:{vxb:+.3f},lat:{vyb:+.3f})  "
              f"face_yaw={np.degrees(face):+5.1f}° Δyaw={np.degrees(d_yaw_face):+5.1f}°")

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img, origin='upper')

    # World axes label (top-left corner)
    u0, v0 = w2p([0, 0], cam_pos, cam_mat, fovy, W, H, fz)
    ux, vx = w2p([0.4, 0], cam_pos, cam_mat, fovy, W, H, fz)
    uy, vy = w2p([0, 0.4], cam_pos, cam_mat, fovy, W, H, fz)
    ax.annotate('', xy=(ux, vx), xytext=(u0, v0), arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax.annotate('', xy=(uy, vy), xytext=(u0, v0), arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    ax.text(ux+5, vx-5, 'W+x', color='red', fontsize=11, fontweight='bold')
    ax.text(uy+5, vy-5, 'W+y', color='blue', fontsize=11, fontweight='bold')

    # Robot star + body axes
    ru, rv = w2p(robot_xy, cam_pos, cam_mat, fovy, W, H, fz)
    ax.plot(ru, rv, '*', color='gold', markersize=28, markeredgecolor='black', markeredgewidth=2, zorder=12)
    L = 0.5
    bx_w = robot_xy + L*np.array([np.cos(yaw), np.sin(yaw)])
    by_w = robot_xy + L*np.array([np.cos(yaw+np.pi/2), np.sin(yaw+np.pi/2)])
    bxu, bxv = w2p(bx_w, cam_pos, cam_mat, fovy, W, H, fz)
    byu, byv = w2p(by_w, cam_pos, cam_mat, fovy, W, H, fz)
    ax.annotate('', xy=(bxu, bxv), xytext=(ru, rv), arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    ax.annotate('', xy=(byu, byv), xytext=(ru, rv), arrowprops=dict(arrowstyle='->', color='gray', lw=3))
    ax.text(bxu+5, bxv-5, 'B+x (fwd)', color='orange', fontsize=11, fontweight='bold')
    ax.text(byu+5, byv-5, 'B+y (left)', color='gray', fontsize=11, fontweight='bold')

    # Waypoints
    prev_uv = (ru, rv)
    for i, wp in enumerate(traj_world):
        wp_xy = np.asarray(wp)
        u, v = w2p(wp_xy, cam_pos, cam_mat, fovy, W, H, fz)
        color = 'red' if i == 0 else ('magenta' if i < len(traj_world)-1 else 'lime')
        size = 16 if i in (0, len(traj_world)-1) else 12
        ax.plot(u, v, 'o', color=color, markersize=size, markeredgecolor='white', markeredgewidth=1.5, zorder=11)
        ax.text(u+10, v+12, str(i), fontsize=11, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        ax.plot([prev_uv[0], u], [prev_uv[1], v], '-', color='magenta', lw=1.2, alpha=0.5)
        prev_uv = (u, v)

    # Goal
    gu, gv = w2p(goal_xy, cam_pos, cam_mat, fovy, W, H, fz)
    ax.plot(gu, gv, 's', color='cyan', markersize=22, markeredgecolor='black', markeredgewidth=2, zorder=12)
    ax.text(gu+15, gv, 'GOAL', color='cyan', fontsize=14, fontweight='bold')

    ax.set_title(
        f"[{label}] waypoints on topview  ({len(traj_world)} wps)\n"
        f"robot=({robot_xy[0]:.2f},{robot_xy[1]:.2f}) yaw={np.degrees(yaw):.0f}°  goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f})  "
        f"RED=wp[0]  MAGENTA=mid  LIME=wp[last]",
        fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{label}_wps.png")
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  saved {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for task, layout, style, label, dump in CASES:
        render(task, layout, style, label, dump)


if __name__ == "__main__":
    main()
