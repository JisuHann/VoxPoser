"""Layout-별 motion audit — 4방 막힘 + 매핑 검증.

각 layout × scenario에서:
  - Reset, 시작 pose 읽기
  - 4 방향 명령 (+v_x, -v_x, +v_y, -v_y) 각각 30 step
  - 결과 world Δ를 body frame (R(π-yaw) inverse)으로 분해
  - "blocked" = ||world Δ|| < 50mm (벽에 막힌 것)
  - 매핑 일관성: 각 명령이 R(π-yaw)에 따른 예측 방향과 일치?

Output: 표 — layout마다 시작 yaw, 각 4방향 모션 결과, blocked 여부, 매핑 OK.
"""
import os
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')

from envs.robocasa_env import VoxPoserRobocasa
from utils.arguments import get_config
from transforms3d.euler import quat2euler

N_STEPS = 30
MAG = 1.0
BLOCK_THR_MM = 50    # below this → blocked

# (task, layout, style, label) — pure-pursuit가 실제 쓰는 Wine RouteA task
SCENES = []
for L in [0, 1, 7]:
    SCENES.append(("NavigateKitchenWineBlockingRouteA", L, 3, f'wine_a_L{L}'))


def read(env):
    obs = env.env._get_observations()
    q = obs['robot0_base_quat']
    yaw = quat2euler([q[3], q[0], q[1], q[2]])[2]
    pos = obs['robot0_base_pos'][:2].copy()
    return float(yaw), pos


def apply_n(env, action, n=N_STEPS):
    yaw0, pos0 = read(env)
    for _ in range(n):
        env.apply_navigation_action(np.array(action))
    yawf, posf = read(env)
    return yaw0, pos0, yawf, posf


def world_to_body_rpiy(yaw, dx_w, dy_w):
    """world Δ → body Δ using R(π-yaw) inverse (correct for our controller)."""
    bx = -dx_w * np.cos(yaw) + dy_w * np.sin(yaw)
    by = -dx_w * np.sin(yaw) - dy_w * np.cos(yaw)
    return bx, by


def predict_world_dir_rpiy(yaw, body_cmd):
    """body cmd → predicted world dir using R(π-yaw)."""
    bx, by = body_cmd
    dx = -np.cos(yaw)*bx - np.sin(yaw)*by
    dy = np.sin(yaw)*bx - np.cos(yaw)*by
    return dx, dy


def run_layout(task, layout, style, label):
    config = get_config(config_path='configs/robocasa_config.yaml', task_type='navigation')
    task_cfg = config['task']
    task_cfg['layout_ids'] = layout
    task_cfg['style_ids'] = style
    env = VoxPoserRobocasa(task_name=task, task_config=task_cfg)
    env.env.reset()
    yaw_init, pos_init = read(env)
    print(f"\n{'='*100}")
    print(f"[{label}]  start pos=({pos_init[0]:+.3f}, {pos_init[1]:+.3f})  yaw={np.degrees(yaw_init):+.1f}°")
    print(f"{'='*100}")
    print(f"  {'cmd':<14} {'world Δ (mm)':<22} {'body Δ (mm)':<22} {'blocked?':<10} {'match expected?':<25}")
    print(f"  {'-'*14} {'-'*22} {'-'*22} {'-'*10} {'-'*25}")
    for action, name in [
        ([+MAG, 0, 0],  '+v_x (body fwd)'),
        ([-MAG, 0, 0],  '-v_x (body bwd)'),
        ([0, +MAG, 0],  '+v_y (body L)  '),
        ([0, -MAG, 0],  '-v_y (body R)  '),
    ]:
        env.env.reset()
        yaw0, p0, yawf, pf = apply_n(env, action)
        dx_w = (pf[0] - p0[0]) * 1000
        dy_w = (pf[1] - p0[1]) * 1000
        bx, by = world_to_body_rpiy(yaw0, dx_w/1000, dy_w/1000)
        bx_mm, by_mm = bx*1000, by*1000
        mag_w = float(np.hypot(dx_w, dy_w))
        blocked = mag_w < BLOCK_THR_MM
        # Predicted body direction for the command
        pred_bx, pred_by = action[0], action[1]
        # Body Δ should have same sign as commanded
        match_bx = "✓" if pred_bx == 0 or (np.sign(bx_mm) == np.sign(pred_bx) and abs(bx_mm) > 50) else "✗" if not blocked else "(blocked)"
        match_by = "✓" if pred_by == 0 or (np.sign(by_mm) == np.sign(pred_by) and abs(by_mm) > 50) else "✗" if not blocked else "(blocked)"
        match_str = f"bx:{match_bx} by:{match_by}"
        print(f"  {name:<14} ({dx_w:+5.0f}, {dy_w:+5.0f})       (fwd:{bx_mm:+5.0f}, lat:{by_mm:+5.0f})  "
              f"{'YES' if blocked else 'no':<10} {match_str:<25}")

    # Pure omega — verify yaw sign
    env.env.reset()
    yaw0, p0, yawf, pf = apply_n(env, [0, 0, +MAG])
    dyaw = (yawf - yaw0 + np.pi) % (2*np.pi) - np.pi
    drift = float(np.linalg.norm(pf - p0)) * 1000
    print(f"  {'+omega       ':<14}  Δyaw={np.degrees(dyaw):+5.1f}°  drift={drift:.0f}mm  "
          f"{'(positive = CCW correct ✓)' if dyaw > 0 else '(✗ wrong sign)'}")


def main():
    for task, layout, style, label in SCENES:
        try:
            run_layout(task, layout, style, label)
        except Exception as e:
            print(f"\n[{label}] FAILED: {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
