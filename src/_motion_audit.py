import numpy as np
from envs.robocasa_env import VoxPoserRobocasa
from utils.arguments import get_config
from transforms3d.euler import quat2euler
config=get_config(config_path='configs/robocasa_config.yaml',task_type='navigation')
tc=config['task']; tc['layout_ids']=0; tc['style_ids']=3
env=VoxPoserRobocasa(task_name='NavigateKitchenWineBlockingRouteA',task_config=tc)
env.env.reset()
def get_yaw():
    q=env.env._get_observations()['robot0_base_quat']
    return float(quat2euler([q[3],q[0],q[1],q[2]])[2])
def run_omega(omega_cmd, n_steps=50):
    env.env.reset()
    y0 = get_yaw()
    yaws=[y0]
    a = np.zeros(env.env.action_dim)
    a[9] = omega_cmd  # omega at index 9 per apply_navigation_action layout
    for _ in range(n_steps):
        env.env.step(a)
        yaws.append(get_yaw())
    rates=[]
    for i in range(1,len(yaws)):
        dy=(yaws[i]-yaws[i-1]+np.pi)%(2*np.pi)-np.pi
        rates.append(dy)
    total=(yaws[-1]-y0+np.pi)%(2*np.pi)-np.pi
    return total, np.mean(rates), max(rates,key=abs)
print(f'action_dim={env.env.action_dim}, omega index=9')
print()
print('omega_cmd  total_dy_deg(50step)  per-step_avg_deg  per-step_max_deg')
for omega in [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]:
    total, avg, mx = run_omega(omega, n_steps=50)
    print(f'  {omega:.2f}    {np.degrees(total):>10.1f}      {np.degrees(avg):>10.3f}      {np.degrees(mx):>10.3f}')
