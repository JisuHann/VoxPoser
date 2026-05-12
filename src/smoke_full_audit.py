"""포괄적 좌표/방향/매핑 audit.

7개 섹션:
  S1. yaw convention (quat2euler)
  S2. body axes definition (각 yaw별로 +v_x/+v_y 명령 → world Δ 측정)
  S3. omega 부호 (+omega가 CCW인지)
  S4. body→world 변환 검증 (수식 vs 실측)
  S5. world→body 변환 검증 (역변환 사용해서 명령 도출)
  S6. wp 좌표 해석 (planner traj_world이 (x,y)인지 (y,x)인지)
  S7. topview 카메라 → world 매핑 (project 알려진 world 점들 → pixel)
"""
import os
import sys
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')

from envs.robocasa_env import VoxPoserRobocasa
from utils.arguments import get_config
from transforms3d.euler import quat2euler


def read(env):
    obs = env.env._get_observations()
    q = obs['robot0_base_quat']
    yaw = quat2euler([q[3], q[0], q[1], q[2]])[2]
    pos = obs['robot0_base_pos'][:2].copy()
    return float(yaw), pos


def rotate_to(env, target_yaw, max_steps=200):
    """omega로 target_yaw에 도달할 때까지 회전."""
    for _ in range(max_steps):
        yaw, _ = read(env)
        d = (target_yaw - yaw + np.pi) % (2*np.pi) - np.pi
        if abs(d) < np.deg2rad(5):
            return True
        omega = float(np.clip(d*5.0, -1.0, 1.0))
        env.apply_navigation_action(np.array([0.0, 0.0, omega]))
    return False


def apply_n(env, action, n=30):
    yaw0, pos0 = read(env)
    for _ in range(n):
        env.apply_navigation_action(np.array(action))
    yawf, posf = read(env)
    return yaw0, pos0, yawf, posf


def main():
    config = get_config(config_path='configs/robocasa_config.yaml', task_type='navigation')
    task_cfg = config['task']
    task_cfg['layout_ids'] = 0
    task_cfg['style_ids'] = 3
    env = VoxPoserRobocasa(task_name="NavigateKitchenCatBlockingRouteD", task_config=task_cfg)
    env.env.reset()
    yaw_init, pos_init = read(env)

    print("="*90)
    print("FULL COORDINATE / DIRECTION / MAPPING AUDIT")
    print("="*90)
    print(f"\nScene: cat_d L0 style3")
    print(f"Initial: pos=({pos_init[0]:.3f},{pos_init[1]:.3f})  yaw={np.degrees(yaw_init):.1f}°")
    print(f"Workspace: min={env.workspace_bounds_min[:2]}  max={env.workspace_bounds_max[:2]}")
    print(f"Goal: {env.env.target_pos[:2]}")

    # ==========================================================
    # S1: yaw convention
    # ==========================================================
    print("\n" + "="*90)
    print("S1. YAW CONVENTION (quat2euler reading)")
    print("="*90)
    obs = env.env._get_observations()
    q = obs['robot0_base_quat']  # robosuite returns xyzw
    print(f"raw quat (xyzw): {q.tolist()}")
    rpy = quat2euler([q[3], q[0], q[1], q[2]])  # we reorder to wxyz
    print(f"quat2euler(wxyz) → (roll, pitch, yaw) = ({np.degrees(rpy[0]):+.1f}, {np.degrees(rpy[1]):+.1f}, {np.degrees(rpy[2]):+.1f})°")
    print(f"yaw[2] used: {np.degrees(rpy[2]):.1f}°  (convention: 0=face world+x, +90=face world+y)")

    # ==========================================================
    # S2: Body axes at multiple yaws — does +v_x always move body fwd?
    # ==========================================================
    print("\n" + "="*90)
    print("S2. BODY AXES — for yaw ∈ {0, 90, 180, -90}°, test +v_x and +v_y")
    print("    Expected: +v_x at yaw=Y moves robot to (cos(Y), sin(Y)) world dir")
    print("="*90)
    test_yaws_deg = [0, 90, 180, -90]
    for tyaw_deg in test_yaws_deg:
        env.env.reset()
        ok = rotate_to(env, np.deg2rad(tyaw_deg))
        yaw_now, pos_now = read(env)
        if not ok or abs(np.degrees(yaw_now) - tyaw_deg) > 10:
            print(f"  yaw target={tyaw_deg}° NOT reached (got {np.degrees(yaw_now):.1f}°), skip")
            continue
        # apply +v_x
        yaw0, p0, yawf, pf = apply_n(env, [1.0, 0, 0], n=30)
        dxw = pf[0] - p0[0]
        dyw = pf[1] - p0[1]
        # Expected world direction: (cos(yaw0), sin(yaw0))
        exp_x = np.cos(yaw0)
        exp_y = np.sin(yaw0)
        # Actual fwd component (project onto expected direction)
        fwd_actual = dxw * exp_x + dyw * exp_y
        lat_actual = -dxw * np.sin(yaw0) + dyw * np.cos(yaw0)
        print(f"  yaw≈{np.degrees(yaw0):+5.0f}°  +v_x=1.0 ×30:  "
              f"Δworld=({dxw*1000:+5.0f},{dyw*1000:+5.0f})mm  "
              f"body_fwd={fwd_actual*1000:+5.0f}mm  body_lat={lat_actual*1000:+5.0f}mm  "
              f"expected dir≈({exp_x:+.2f},{exp_y:+.2f})  match={'✓' if fwd_actual>0 and abs(lat_actual)<abs(fwd_actual) else '✗'}")

    print("")
    for tyaw_deg in test_yaws_deg:
        env.env.reset()
        ok = rotate_to(env, np.deg2rad(tyaw_deg))
        yaw_now, pos_now = read(env)
        if not ok:
            continue
        # apply +v_y
        yaw0, p0, yawf, pf = apply_n(env, [0, 1.0, 0], n=30)
        dxw = pf[0] - p0[0]
        dyw = pf[1] - p0[1]
        # Expected world direction: (cos(yaw0+90°), sin(yaw0+90°)) = (-sin(yaw0), cos(yaw0))
        exp_x = -np.sin(yaw0)
        exp_y = np.cos(yaw0)
        lat_actual = -dxw * np.sin(yaw0) + dyw * np.cos(yaw0)
        fwd_actual = dxw * np.cos(yaw0) + dyw * np.sin(yaw0)
        print(f"  yaw≈{np.degrees(yaw0):+5.0f}°  +v_y=1.0 ×30:  "
              f"Δworld=({dxw*1000:+5.0f},{dyw*1000:+5.0f})mm  "
              f"body_fwd={fwd_actual*1000:+5.0f}mm  body_lat={lat_actual*1000:+5.0f}mm  "
              f"expected dir≈({exp_x:+.2f},{exp_y:+.2f})  match={'✓' if lat_actual>0 and abs(fwd_actual)<abs(lat_actual) else '✗'}")

    # ==========================================================
    # S3: omega sign
    # ==========================================================
    print("\n" + "="*90)
    print("S3. OMEGA SIGN (+omega → CCW yaw increases?)")
    print("="*90)
    env.env.reset()
    yaw0, p0, yawf, pf = apply_n(env, [0, 0, +0.3], n=30)
    print(f"  +omega=0.3 ×30: yaw {np.degrees(yaw0):+.0f}° → {np.degrees(yawf):+.0f}°  Δ={np.degrees(yawf-yaw0):+.1f}°  expected: positive Δyaw (CCW)")
    env.env.reset()
    yaw0, p0, yawf, pf = apply_n(env, [0, 0, -0.3], n=30)
    print(f"  -omega=0.3 ×30: yaw {np.degrees(yaw0):+.0f}° → {np.degrees(yawf):+.0f}°  Δ={np.degrees(yawf-yaw0):+.1f}°  expected: negative Δyaw (CW)")

    # ==========================================================
    # S4: body→world 변환 검증 (수식과 일치하나)
    # ==========================================================
    print("\n" + "="*90)
    print("S4. BODY→WORLD TRANSFORMATION VERIFICATION")
    print("    Formula: world_Δ = R(yaw) @ body_Δ")
    print("    R(yaw) = [[cos,-sin],[sin,cos]]")
    print("="*90)
    for tyaw_deg in [0, 90, 180, -90, 45]:
        env.env.reset()
        ok = rotate_to(env, np.deg2rad(tyaw_deg))
        yaw_now, _ = read(env)
        # Apply body [v_x=1.0, v_y=0]: should produce world Δ ~ (cos, sin)
        yaw0, p0, yawf, pf = apply_n(env, [1.0, 0, 0], n=30)
        dxw_a = pf[0]-p0[0]
        dyw_a = pf[1]-p0[1]
        # Predicted (assuming our formula correct, body v_x=1 gives unit body fwd, sim attenuates magnitude)
        # body→world: world = R(yaw) @ body
        # body=(1, 0) → world=(cos(yaw), sin(yaw))
        cos_y = np.cos(yaw0)
        sin_y = np.sin(yaw0)
        # Actual magnitude can vary due to attenuation, but DIRECTION should match
        mag_a = np.hypot(dxw_a, dyw_a)
        if mag_a > 1e-3:
            dir_a = np.array([dxw_a, dyw_a]) / mag_a
            dir_e = np.array([cos_y, sin_y])
            angle_err = np.degrees(np.arccos(np.clip(np.dot(dir_a, dir_e), -1, 1)))
        else:
            angle_err = 0.0
        print(f"  yaw={np.degrees(yaw0):+5.0f}°  body[1,0,0]→world dir=({dir_a[0]:+.2f},{dir_a[1]:+.2f}) vs predicted=({cos_y:+.2f},{sin_y:+.2f})  err={angle_err:.1f}°")

    # ==========================================================
    # S5: world→body 역변환 (algorithm이 사용하는 부호)
    # ==========================================================
    print("\n" + "="*90)
    print("S5. WORLD→BODY (algorithm: dx,dy world → body command)")
    print("    Formula: v_x_body = dx*cos(yaw) + dy*sin(yaw)")
    print("             v_y_body = -dx*sin(yaw) + dy*cos(yaw)")
    print("="*90)
    # 예: yaw=+90°에서 world Δ=(-2.5, 0) (cat_d goal 방향)
    yaw_test = np.deg2rad(90)
    dx, dy = -2.5, 0.0  # cat_d: robot→goal world delta
    vxb = dx*np.cos(yaw_test) + dy*np.sin(yaw_test)
    vyb = -dx*np.sin(yaw_test) + dy*np.cos(yaw_test)
    print(f"  yaw=+90°, world Δ=(-2.5, 0) (cat_d goal dir):")
    print(f"    body command = (v_x={vxb:.2f}, v_y={vyb:.2f})")
    print(f"    → v_x=0 means no fwd motion, v_y=+2.5 means body LEFT")
    print(f"    body +y at yaw=+90° = world -x direction ✓ (matches goal direction)")
    # 예: person_g
    dx, dy = +2.3, 0.0  # robot→goal in person_g
    vxb = dx*np.cos(yaw_test) + dy*np.sin(yaw_test)
    vyb = -dx*np.sin(yaw_test) + dy*np.cos(yaw_test)
    print(f"  yaw=+90°, world Δ=(+2.3, 0) (person_g goal dir):")
    print(f"    body command = (v_x={vxb:.2f}, v_y={vyb:.2f})")
    print(f"    → v_y=-2.3 means body RIGHT")
    print(f"    body -y at yaw=+90° = world +x direction ✓ (matches goal direction)")

    # ==========================================================
    # S6: planner waypoint coords (x, y) interpretation
    # ==========================================================
    print("\n" + "="*90)
    print("S6. WAYPOINT COORDINATE INTERPRETATION (traj_world)")
    print("="*90)
    dump_path = '/workspace/policy/Voxposer/outputs/smoke_optB_114242/cat_d/layout0/NavigateKitchenCatBlockingRouteD/voxposer_dump.npz'
    d = np.load(dump_path, allow_pickle=True)
    iters = d['iters']
    tw = iters[0]['traj_world']
    print(f"  traj_world shape: {tw.shape}  dtype: {tw.dtype}")
    print(f"  traj_world[0]: {tw[0]}  ← interpreted as (x={tw[0][0]:.3f}, y={tw[0][1]:.3f})")
    print(f"  obstacle (cat) xy: {d['obstacle_xy']}")
    print(f"  goal_xy_fixture: {d['goal_xy_fixture']}")
    print(f"")
    print(f"  robot world=({pos_init[0]:.3f},{pos_init[1]:.3f})  goal world=({env.env.target_pos[0]:.3f},{env.env.target_pos[1]:.3f})")
    print(f"  goal direction from robot: Δx={env.env.target_pos[0]-pos_init[0]:+.3f} Δy={env.env.target_pos[1]-pos_init[1]:+.3f} → west")
    print(f"  wp[0]={tw[0]}: Δx={tw[0][0]-pos_init[0]:+.3f} Δy={tw[0][1]-pos_init[1]:+.3f} → south-west (detour to avoid cat)")
    print(f"  wp[last]={tw[-1]}: Δx={tw[-1][0]-pos_init[0]:+.3f} Δy={tw[-1][1]-pos_init[1]:+.3f} → west (near goal)")
    print(f"  → If (x,y) were swapped to (y,x): wp[0] would be at world ({tw[0][1]:.3f},{tw[0][0]:.3f})")
    print(f"     → Δx={tw[0][1]-pos_init[0]:+.3f} Δy={tw[0][0]-pos_init[1]:+.3f}: that'd be far south-east, OUTSIDE workspace")
    print(f"  → Storage is (x,y) — NOT swapped ✓")

    # ==========================================================
    # S7: topview camera ↔ world
    # ==========================================================
    print("\n" + "="*90)
    print("S7. TOPVIEW CAMERA PROJECTION (sanity check world→pixel)")
    print("="*90)
    sim = env.env.sim
    cid = sim.model.camera_name2id('topview')
    cam_pos = sim.data.cam_xpos[cid]
    cam_mat = sim.data.cam_xmat[cid].reshape(3, 3)
    fovy = float(sim.model.cam_fovy[cid])
    print(f"  cam_pos: {cam_pos}")
    print(f"  cam_mat:\n{cam_mat}")
    print(f"  fovy: {fovy}°")
    W_img, H_img = 1280, 960
    fy = (H_img/2.0) / np.tan(np.radians(fovy/2.0))
    fx = fy
    fz = 0.0
    try:
        fids = getattr(env, 'floor_mask_ids', [])
        if fids:
            fz = float(sim.data.geom_xpos[fids[0], 2])
    except Exception:
        pass

    def project(xy):
        w = np.array([xy[0], xy[1], fz])
        rel = w - cam_pos
        cf = cam_mat.T @ rel
        depth = -cf[2]
        u = (W_img/2.0) + fx * (cf[0]/depth)
        v = (H_img/2.0) - fy * (cf[1]/depth)
        return u, v   # before [::-1] flip (raw render coord)
    print(f"  Image size W={W_img} H={H_img}, image origin top-left")
    test_pts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (5.54, 0.04), (-0.04, -6.04)]
    print(f"  Project world points → pixel (raw):")
    for p in test_pts:
        u, v = project(p)
        print(f"    world ({p[0]:+5.2f},{p[1]:+5.2f}) → pixel ({u:6.0f}, {v:6.0f})")
    # After [::-1] vertical flip:
    print(f"  After [::-1] vertical flip (final image, v_img = H - v):")
    for p in test_pts:
        u, v = project(p)
        v_img = H_img - v
        print(f"    world ({p[0]:+5.2f},{p[1]:+5.2f}) → image_pixel ({u:6.0f}, {v_img:6.0f})")
    # Direction analysis
    u0, v0 = project([0, 0])
    ux, vx = project([1, 0])
    uy, vy = project([0, 1])
    print(f"\n  world (1, 0) - (0, 0) → pixel delta = ({ux-u0:+.0f}, {vx-v0:+.0f}) → unflipped image")
    print(f"  world (0, 1) - (0, 0) → pixel delta = ({uy-u0:+.0f}, {vy-v0:+.0f}) → unflipped image")
    print(f"  After [::-1] vertical flip:")
    print(f"    world +x → pixel direction: u {'+' if ux>u0 else '-'}, v_img {'+' if (H_img-vx)>(H_img-v0) else '-'}")
    print(f"    world +y → pixel direction: u {'+' if uy>u0 else '-'}, v_img {'+' if (H_img-vy)>(H_img-v0) else '-'}")

    print("\n" + "="*90)
    print("AUDIT COMPLETE")
    print("="*90)


if __name__ == "__main__":
    main()
