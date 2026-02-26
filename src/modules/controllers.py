import numpy as np
from transforms3d.quaternions import mat2quat
from utils.utils import normalize_vector
import copy
import time
from modules.dynamics_models import PushingDynamicsModel

EE_ALIAS = ['ee', 'endeffector', 'end_effector', 'end effector', 'gripper', 'hand']


class NavigationController:
    def __init__(self, env, config):
        self.config = config
        self.env = env

    def execute(self, waypoint):
        """
        Execute a navigation waypoint.
        :param waypoint: [target_xy, target_rotation, target_velocity]
        :return: info dict with mp_info
        """
        info = dict()
        result = self.env.apply_navigation_action(waypoint)
        info['mp_info'] = result
        return info


class ManipulationController:
    def __init__(self, env, config):
        self.config = config
        self.env = env
        self.dynamics_model = PushingDynamicsModel()

    def _calculate_ee_rot(self, pushing_dir):
        """
        Given a pushing direction, calculate the rotation matrix for the end effector.
        Slanted towards table for safer interaction.
        """
        pushing_dir = normalize_vector(pushing_dir)
        desired_dir = pushing_dir + np.array([0, 0, -np.linalg.norm(pushing_dir)])
        desired_dir = normalize_vector(desired_dir)
        left = np.cross(pushing_dir, desired_dir)
        left = normalize_vector(left)
        up = np.array(desired_dir, dtype=np.float32)
        up = normalize_vector(up)
        forward = np.cross(left, up)
        forward = normalize_vector(forward)
        rotmat = np.eye(3).astype(np.float32)
        rotmat[:3, 0] = forward
        rotmat[:3, 1] = left
        rotmat[:3, 2] = up
        quat_wxyz = mat2quat(rotmat)
        return quat_wxyz

    def _apply_mpc_control(self, control, target_velocity=1):
        """Apply MPC control to push the object."""
        contact_position = control[:3]
        pushing_dir = control[3:6]
        pushing_dist = control[6]
        ee_quat = self._calculate_ee_rot(pushing_dir)
        start_dist = 0.08
        t_start = contact_position - pushing_dir * start_dist
        t_interact = contact_position + pushing_dir * pushing_dist
        t_rest = contact_position - pushing_dir * start_dist * 0.8

        self.env.close_gripper()
        self.env.move_to_pose(np.concatenate([t_start, ee_quat]), speed=target_velocity)
        print('[controllers.py] moved to start pose', end='; ')
        self.env.move_to_pose(np.concatenate([t_interact, ee_quat]), speed=target_velocity * 0.2)
        print('[controllers.py] moved to final pose', end='; ')
        self.env.move_to_pose(np.concatenate([t_rest, ee_quat]), speed=target_velocity * 0.33)
        print('[controllers.py] back to release pose', end='; ')
        self.env.reset_to_default_pose()
        print('[controllers.py] back to default pose')

    def execute(self, movable_obs, waypoint):
        """
        Execute a manipulation waypoint.
        If movable is end effector, move directly. If object, use MPC with dynamics model.

        :param movable_obs: observation dict of the object to be moved
        :param waypoint: [target_xyz, target_rotation, target_velocity, target_gripper]
        :return: info dict
        """
        info = dict()
        target_xyz, target_rotation, target_velocity, target_gripper = waypoint
        object_centric = (not movable_obs['name'].lower() in EE_ALIAS)
        if not object_centric:
            target_pose = np.concatenate([target_xyz, target_rotation])
            result = self.env.apply_action(np.concatenate([target_pose, [target_gripper]]))
            info['mp_info'] = result
        else:
            start = time.time()
            movable_obs = {key: value for key, value in movable_obs.items() if key in ['_point_cloud_world']}
            best_control, self.mpc_info = self.random_shooting_MPC(movable_obs, target_xyz)
            print('[controllers.py] mpc search completed in {} seconds with {} samples'.format(time.time() - start, self.config.num_samples))
            self.mpc_velocity = target_velocity
            self._apply_mpc_control(best_control[0])
            print(f'[controllers.py] applied control (pos: {best_control[0][:3].round(4)}, dir: {best_control[0][3:6].round(4)}, dist: {best_control[0][6:].round(4)})')
            info['mpc_info'] = self.mpc_info
            info['mpc_control'] = best_control[0]
        return info

    def random_shooting_MPC(self, start_obs, target):
        obs_sequences = []
        controls_sequences = []
        info = dict()
        batched_start_obs = {}
        for key, value in start_obs.items():
            batched_start_obs[key] = np.repeat(value[None, ...], self.config.num_samples, axis=0)
        obs_sequences.append(batched_start_obs)
        for t in range(self.config.horizon_length):
            curr_obs = copy.deepcopy(obs_sequences[-1])
            controls = self.generate_random_control(curr_obs, target)
            pred_next_obs = self.forward_step(curr_obs, controls)
            obs_sequences.append(pred_next_obs)
            controls_sequences.append(controls)
        costs = self.calculate_cost(obs_sequences, controls_sequences, target)
        best_traj_idx = np.argmin(costs)
        best_controls_sequence = np.array([control_per_step[best_traj_idx] for control_per_step in controls_sequences])
        info['best_controls_sequence'] = best_controls_sequence
        info['best_cost'] = costs[best_traj_idx]
        info['costs'] = costs
        info['controls_sequences'] = controls_sequences
        info['obs_sequences'] = obs_sequences
        return best_controls_sequence, info

    def forward_step(self, obs, controls):
        """
        Forward dynamics simulation.
        obs: dict with point cloud [B, N, 3]
        controls: [B, 7]
        returns: predicted next obs
        """
        pcs = obs['_point_cloud_world']
        contact_position = controls[:, :3]
        pushing_dir = controls[:, 3:6]
        pushing_dist = controls[:, 6:]
        inputs = (pcs, contact_position, pushing_dir, pushing_dist)
        next_pcs = self.dynamics_model.forward(inputs)
        next_obs = copy.deepcopy(obs)
        next_obs['_point_cloud_world'] = next_pcs
        return next_obs

    def generate_random_control(self, obs, target):
        """
        Sample random controls: contact position on point cloud, pushing direction towards target, random distance.
        returns: [B, 7]
        """
        pcs = obs['_point_cloud_world']
        num_samples, num_points, _ = pcs.shape
        points_idx = np.random.randint(0, num_points, num_samples)
        contact_positions = pcs[np.arange(num_samples), points_idx]
        pushing_dirs = normalize_vector(target - contact_positions)
        pushing_dist = np.random.uniform(-0.02, 0.09, size=(num_samples, 1))
        controls = np.concatenate([contact_positions, pushing_dirs, pushing_dist], axis=1)
        return controls

    def calculate_cost(self, obs_sequences, controls_sequences, target_xyz):
        """
        Cost = L2 distance from predicted final object position to target.
        returns: [B]
        """
        num_samples, _, _ = obs_sequences[0]['_point_cloud_world'].shape
        last_obs = obs_sequences[-1]
        costs = []
        for i in range(num_samples):
            last_pc = last_obs['_point_cloud_world'][i]
            last_position = np.mean(last_pc, axis=0)
            cost = np.linalg.norm(last_position - target_xyz)
            costs.append(cost)
        costs = np.array(costs)
        return costs
