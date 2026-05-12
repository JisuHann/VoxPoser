"""Overlay 4 coordinate frames on a single topview frame.

For each scene (cat_d, person_g) at start state:
  - W (World): red +x arrow, blue +y arrow, origin marker
  - G (Planner Grid): 4 corners (workspace_bounds projected to pixel) + grid axis labels
  - P (Topview Pixel): u/v axes in corner
  - B (Body): gold +x (fwd), gray +y (left) at robot pose
  - markers: robot star, goal square

Saved to /workspace/tmp/axes_overlay/<scene>.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault('MUJOCO_GL', 'egl')

from envs.robocasa_env import VoxPoserRobocasa
from utils.arguments import get_config
from transforms3d.euler import quat2euler

OUT_DIR = '/workspace/tmp/axes_overlay'
W, H = 1280, 960   # high-res for inspection

SCENES = [
    ("NavigateKitchenCatBlockingRouteD", 0, 3, 'cat_d'),
    ("NavigateKitchenHumanBlockingRouteG", 0, 3, 'person_g'),
]


def read(env):
    obs = env.env._get_observations()
    q = obs['robot0_base_quat']
    yaw = quat2euler([q[3], q[0], q[1], q[2]])[2]
    pos = obs['robot0_base_pos'][:2].copy()
    return float(yaw), pos


def world_to_pixel(world_xy, cam_pos, cam_mat, fovy, W, H, floor_z=0.0):
    """Project a single world point (x,y,floor_z) into topview pixel (u, v)."""
    fy = (H / 2.0) / np.tan(np.radians(fovy / 2.0))
    fx = fy
    w = np.array([world_xy[0], world_xy[1], floor_z])
    rel = w - cam_pos
    cam_frame = cam_mat.T @ rel
    depth = -cam_frame[2]
    u = (W / 2.0) + fx * (cam_frame[0] / depth)
    v = (H / 2.0) - fy * (cam_frame[1] / depth)
    return float(u), float(v)


def render_overlay(env, label, out_path):
    sim = env.env.sim
    yaw, robot_xy = read(env)
    goal_xy = np.array(env.env.target_pos[:2])
    ws_min = np.asarray(env.workspace_bounds_min)
    ws_max = np.asarray(env.workspace_bounds_max)

    # Camera params
    cid = sim.model.camera_name2id('topview')
    cam_pos = sim.data.cam_xpos[cid].copy()
    cam_mat = sim.data.cam_xmat[cid].reshape(3, 3).copy()
    fovy = float(sim.model.cam_fovy[cid])

    # Floor z
    floor_z = 0.0
    try:
        fids = getattr(env, 'floor_mask_ids', [])
        if fids:
            floor_z = float(sim.data.geom_xpos[fids[0], 2])
    except Exception:
        pass

    # Render topview (NOTE: sim.render returns vertically flipped; we use [::-1])
    img = sim.render(camera_name='topview', width=W, height=H)[::-1]

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img, origin='upper')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # match imshow upper origin

    # === W axes: draw at world origin (0, 0) ===
    u0, v0 = world_to_pixel([0, 0], cam_pos, cam_mat, fovy, W, H, floor_z)
    # Note: sim.render is flipped → after [::-1], v_in_image = H - v_proj
    v0_img = H - v0
    ax.plot(u0, v0_img, 'k+', markersize=18, markeredgewidth=3)
    # +x world direction: project (0.5, 0) and (0, 0.5)
    ux, vx = world_to_pixel([0.5, 0], cam_pos, cam_mat, fovy, W, H, floor_z)
    uy, vy = world_to_pixel([0, 0.5], cam_pos, cam_mat, fovy, W, H, floor_z)
    vx_img = H - vx
    vy_img = H - vy
    ax.annotate('', xy=(ux, vx_img), xytext=(u0, v0_img),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax.annotate('', xy=(uy, vy_img), xytext=(u0, v0_img),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax.text(ux+5, vx_img-5, 'W +x', color='red', fontsize=14, fontweight='bold')
    ax.text(uy+5, vy_img-5, 'W +y', color='blue', fontsize=14, fontweight='bold')

    # === G corners (workspace_bounds 4 corners) ===
    corners = [
        (ws_min[0], ws_min[1], '(0,0)'),
        (ws_min[0], ws_max[1], f'(0, {env.map_w-1})'),
        (ws_max[0], ws_max[1], f'({env.map_h-1}, {env.map_w-1})'),
        (ws_max[0], ws_min[1], f'({env.map_h-1}, 0)'),
    ]
    cu = []
    cv = []
    for cx, cy, lbl in corners:
        u, v = world_to_pixel([cx, cy], cam_pos, cam_mat, fovy, W, H, floor_z)
        v_img = H - v
        cu.append(u); cv.append(v_img)
        ax.plot(u, v_img, 'o', color='magenta', markersize=10, markeredgecolor='white', markeredgewidth=1.5)
        ax.text(u+8, v_img+8, lbl, color='magenta', fontsize=10, fontweight='bold')
    # connect corners
    cu.append(cu[0]); cv.append(cv[0])
    ax.plot(cu, cv, '--', color='magenta', lw=1.5, alpha=0.7)

    # === Robot pose: B axes ===
    ru, rv = world_to_pixel(robot_xy, cam_pos, cam_mat, fovy, W, H, floor_z)
    rv_img = H - rv
    ax.plot(ru, rv_img, '*', color='gold', markersize=28,
            markeredgecolor='black', markeredgewidth=2, zorder=10)
    # body +x: world direction = (cos(yaw), sin(yaw))
    L = 0.4
    bx_w = robot_xy + L * np.array([np.cos(yaw), np.sin(yaw)])
    by_w = robot_xy + L * np.array([np.cos(yaw + np.pi/2), np.sin(yaw + np.pi/2)])
    bxu, bxv = world_to_pixel(bx_w, cam_pos, cam_mat, fovy, W, H, floor_z)
    byu, byv = world_to_pixel(by_w, cam_pos, cam_mat, fovy, W, H, floor_z)
    bxv_img = H - bxv
    byv_img = H - byv
    ax.annotate('', xy=(bxu, bxv_img), xytext=(ru, rv_img),
                arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    ax.annotate('', xy=(byu, byv_img), xytext=(ru, rv_img),
                arrowprops=dict(arrowstyle='->', color='gray', lw=3))
    ax.text(bxu+5, bxv_img-15, 'B +x (fwd)', color='orange', fontsize=12, fontweight='bold')
    ax.text(byu+5, byv_img-15, 'B +y (left)', color='gray', fontsize=12, fontweight='bold')

    # === Goal ===
    gu, gv = world_to_pixel(goal_xy, cam_pos, cam_mat, fovy, W, H, floor_z)
    gv_img = H - gv
    ax.plot(gu, gv_img, 's', color='cyan', markersize=22,
            markeredgecolor='black', markeredgewidth=2, zorder=10)
    ax.text(gu+15, gv_img, 'goal', color='cyan', fontsize=14, fontweight='bold')

    # === P (topview pixel) axes label ===
    ax.annotate('', xy=(80, 30), xytext=(20, 30),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(20, 90), xytext=(20, 30),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(85, 35, 'P u', fontsize=11, fontweight='bold')
    ax.text(25, 95, 'P v', fontsize=11, fontweight='bold')

    ax.set_title(
        f"[{label}]  axes overlay\n"
        f"robot=({robot_xy[0]:.2f},{robot_xy[1]:.2f}) yaw={np.degrees(yaw):.0f}°  "
        f"goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f})  "
        f"ws_min=({ws_min[0]:.2f},{ws_min[1]:.2f}) ws_max=({ws_max[0]:.2f},{ws_max[1]:.2f})",
        fontsize=12
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")
    print(f"  robot=({robot_xy[0]:.3f},{robot_xy[1]:.3f}) yaw={np.degrees(yaw):.1f}° goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f})")
    print(f"  ws_min={ws_min[:2]} ws_max={ws_max[:2]}")
    print(f"  map_h={env.map_h} map_w={env.map_w}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for task, layout, style, label in SCENES:
        print(f"\n=== {label} ===")
        config = get_config(config_path='configs/robocasa_config.yaml', task_type='navigation')
        task_cfg = config['task']
        task_cfg['layout_ids'] = layout
        task_cfg['style_ids'] = style
        env = VoxPoserRobocasa(task_name=task, task_config=task_cfg)
        env.env.reset()
        render_overlay(env, label, os.path.join(OUT_DIR, f"{label}.png"))


if __name__ == "__main__":
    main()
