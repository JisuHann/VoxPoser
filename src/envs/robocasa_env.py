import os
import numpy as np
import open3d as o3d
import json
from utils.utils import normalize_vector, bcolors
import robosuite
import robocasa
from robosuite import load_composite_controller_config

TABLE_ALIAS =["table","gripper0", 'right', "cutting", "light", "window", 'counter', "floor", 'cab', 'stack', 'wall', 'mobilebase0', 'utensil',"robot0"]

class VoxPoserRobocasa():
    def __init__(self, task_name = "", task_config=None, visualizer=None):
        """
        Initializes the VoxPoserRLBench environment.

        Args:
            visualizer: Visualization interface, optional.
        """
        
        self.task_name = task_name
        self.controller_config = load_composite_controller_config(
            controller="BASIC",
            robot=task_config['robot'],
        )
        # Create argument configuration
        self.env_config = {
            "env_name": task_name,
            "robots": task_config['robot'],
            "controller_configs": self.controller_config,
            "layout_ids": task_config['layout_ids'],
            "style_ids": task_config['style_ids']
        }
        self.offscreen_render = task_config['offscreen_render']
        if self.offscreen_render:
            self.env = robosuite.make(
                **self.env_config,
                has_renderer=False,
                has_offscreen_renderer=True,
                camera_names=task_config['camera_names'],
                camera_widths=task_config['camera_widths'],
                camera_heights=task_config['camera_heights'],
                use_camera_obs=True,
                control_freq=20,
                translucent_robot=False,
            )
        else:
            self.env = robosuite.make(
                **self.env_config,
                has_renderer=True,
                has_offscreen_renderer=False,
                render_camera="robot0_frontview",
                ignore_done=True,
                use_camera_obs=False,
                control_freq=20,
                renderer="mjviewer",
                translucent_robot=True,
            )

        self.camera_names = [self.env.sim.model.camera_id2name(i) for i in range(self.env.sim.model.ncam)]
        forward_vector = np.array([0, 0, 1])
        self.lookat_vectors = {}
        for cam_idx, cam_name in enumerate(self.camera_names):
            extrinsics = self.env.sim.data._data.cam_xmat[cam_idx].reshape(3, 3)
            lookat = extrinsics[:3, :3] @ forward_vector
            self.lookat_vectors[cam_name] = normalize_vector(lookat)
        self._reset_task_variables()

        # workspace variable
        print("TODO(jshan): set workspace bounds")
        # self.workspace_bounds_min = np.array([self.rlbench_env._scene._workspace_minx, self.rlbench_env._scene._workspace_miny, self.rlbench_env._scene._workspace_minz])
        # self.workspace_bounds_max = np.array([self.rlbench_env._scene._workspace_maxx, self.rlbench_env._scene._workspace_maxy, self.rlbench_env._scene._workspace_maxz])

        self.cam_height = task_config['camera_heights']
        self.cam_width = task_config['camera_widths']
        self.map_size = 512
        self._reset_task_variables()

    def get_visible_object_names(self):
        if self.offscreen_render:
            visible_objects = []
            for cam in self.camera_names:
                seg = self.env.sim.render(camera_name=cam, height=self.map_size, width=self.map_size, segmentation=True)[:,:,1]
                visible_geom_ids = np.unique(seg)
                visible_objs = set([self.env.sim.model.body_id2name(self.env.sim.model.geom_bodyid[int(gid)]).split("_")[0] for gid in visible_geom_ids if gid >= 0 and self.env.sim.model.geom_id2name(int(gid)) is not None])
                visible_objects.extend(visible_objs)
            visible_objects = list(set(visible_objects))
            visible_objects = [obj for obj in visible_objects if obj not in TABLE_ALIAS]
        else:
            visible_objects = list(self.env.objects.keys())
        print("Visible objects:", visible_objects)
        return visible_objects
    
    def load_task(self):
        self._reset_task_variables()
        self.reset()

        self.objects = self.get_visible_object_names()
        self.name2ids = {k:[] for k in self.objects}
        for i in range(self.env.sim.model.ngeom):
            name = self.env.sim.model.geom_id2name(i)
            for obj in self.objects:
                if obj in name:
                    self.name2ids[obj].append(i)
                    
    def update_latest_obs(self):
        """
        Docstring for update_latest_obs
        
        :param self: Description
        :param require_pc: Description

        Update point_cloud and mask
        """
        self.latest_obs = self.env._get_observations()
        for cam_name in self.camera_names:
            point_cloud = self.fetch_cam_info(cam_name, require_pc=True, require_mask=False)
            self.latest_obs[f"{cam_name}_point_cloud"] = point_cloud
            mask = self.fetch_cam_info(cam_name, require_pc=False, require_mask=True)
            self.latest_obs[f"{cam_name}_mask"] = mask
            
    def fetch_cam_info(self, cam_name, require_pc=True, require_mask=False):
        cam_config = {
            "height": self.cam_height,
            "width": self.cam_width,
            "depth": require_pc,
            "segmentation": require_mask
        }
        if require_pc:
            output = self.fetch_3d_point_cloud(cam_name, cam_config)
        if require_mask:
            output = self.env.sim.render(camera_name=cam_name, **cam_config)
        return output
    
    def fetch_3d_point_cloud(self, cam_name, cam_config):
        
        cam_id = self.env.sim.model.cam(cam_name).id
        width = cam_config["width"]
        height = cam_config["height"]
        data = self.env.sim.data
        model = self.env.sim.model

        # Intrinsic
        fov = model.cam_fovy[cam_id]
        theta = np.deg2rad(fov)
        fy = height / 2 / np.tan(theta / 2)
        fx = fy
        cx = width / 2
        cy = height / 2

        # Render
        rgb, depth_raw = self.env.sim.render(camera_name=cam_name, **cam_config)
        
        extent = model.stat.extent
        near = model.vis.map.znear * extent
        far = model.vis.map.zfar * extent
        depth = far * near / (far - depth_raw * (far - near))
        
        # 상하 뒤집기
        rgb = rgb[::-1]
        depth = depth[::-1]

        # Open3D로 point cloud 생성
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, 
            depth_scale=1.0, 
            depth_trunc=20.0, 
            convert_rgb_to_intensity=False
        )
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # 카메라 좌표계 점들
        points_cam = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Open3D -> MuJoCo 카메라 좌표계 변환
        # Open3D: X-right, Y-down, Z-forward
        # MuJoCo: X-right, Y-down, -Z-forward
        points_cam[:, 1] = -points_cam[:, 1]  # Y 반전 추가
        points_cam[:, 2] = -points_cam[:, 2]
        
        # Extrinsic: camera to world
        cam_pos = data.cam_xpos[cam_id]
        cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
        
        # 월드 좌표로 변환
        points_world = (cam_rot @ points_cam.T).T + cam_pos
        
        return np.hstack([points_world, colors])
    
    def fetch_obj_segmentation(self, cam_name, query_name):
        seg = self.env.sim.render(camera_name=cam_name, height=self.cam_height, width=self.cam_width, segmentation=True)[:,:,1]
        geom_ids = [self.env.sim.model.geom_name2id(geom_name) \
                                for geom_name in self.env.sim.model.geom_names \
                                if query_name in geom_name]
        obj_geom_mask = np.isin(seg, geom_ids)
        return obj_geom_mask

    def get_3d_obs_by_name(self, query_name):
        """
        Retrieves 3D point cloud observations and normals of an object by its name.

        Args:
            query_name (str): The name of the object to query.

        Returns:
            tuple: A tuple containing object points and object normals.
        """
        assert query_name in self.name2ids, f"Unknown object name: {query_name}"
        obj_ids = self.name2ids[query_name]
        # print(query_name, obj_ids)
        # gather points and masks from all cameras
        self.update_latest_obs()
        points, colors, masks, normals = [], [], [], []
        for cam in self.camera_names:
            points.append(self.latest_obs[f"{cam}_point_cloud"][:,:3].reshape(-1, 3))
            colors.append(self.latest_obs[f"{cam}_point_cloud"][:,3:].reshape(-1, 3))
            masks.append(self.latest_obs[f"{cam}_mask"][:,:,1][::-1].reshape(-1))
            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[-1])
            pcd.colors = o3d.utility.Vector3dVector(colors[-1])
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)
            # use lookat vector to adjust normal vectors TODO(jshan) not sur
            flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)

        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        masks = np.concatenate(masks, axis=0)
        normals = np.concatenate(normals, axis=0)
    
        dist = np.array([points[:,0].max() - points[:,0].min(), 
                         points[:,1].max() - points[:,1].min(),
                         points[:, 2].max()- points[:,2].min(),]).min()
        self.workspace_bounds_min = self.latest_obs['robot0_eef_pos'] - dist/2
        self.workspace_bounds_max = self.latest_obs['robot0_eef_pos'] + dist/2

        # get object points
        obj_points = points[np.isin(masks, obj_ids)]
        if len(obj_points) == 0:
            raise ValueError(f"Object {query_name} not found in the scene")
        obj_colors = colors[np.isin(masks, obj_ids)]
        obj_normals = normals[np.isin(masks, obj_ids)]
        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.colors = o3d.utility.Vector3dVector(obj_colors)
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)
        return (points, normals), (obj_points, obj_normals)

    def visualize_image(rgb):
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()
        return 
    
    def visualize_3d_space(self, xyz, rgb=None):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis.add_geometry(pcd)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
        vis.add_geometry(frame)
        vis.run()
        return

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.
            ignore_grasped_obj (bool): Whether to ignore points corresponding to grasped objects.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        points, colors, masks = [], [], []
        for cam in self.camera_names:
            points.append(getattr(self.latest_obs, f"{cam}_point_cloud").reshape(-1, 3))
            colors.append(getattr(self.latest_obs, f"{cam}_rgb").reshape(-1, 3))
            masks.append(getattr(self.latest_obs, f"{cam}_mask").reshape(-1))
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        masks = np.concatenate(masks, axis=0)

        breakpoint()

        # only keep points within workspace
        chosen_idx_x = (points[:, 0] > self.workspace_bounds_min[0]) & (points[:, 0] < self.workspace_bounds_max[0])
        chosen_idx_y = (points[:, 1] > self.workspace_bounds_min[1]) & (points[:, 1] < self.workspace_bounds_max[1])
        chosen_idx_z = (points[:, 2] > self.workspace_bounds_min[2]) & (points[:, 2] < self.workspace_bounds_max[2])
        points = points[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        colors = colors[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        masks = masks[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]

        if ignore_robot:
            robot_mask = np.isin(masks, self.robot_mask_ids)
            points = points[~robot_mask]
            colors = colors[~robot_mask]
            masks = masks[~robot_mask]
        if self.grasped_obj_ids and ignore_grasped_obj:
            grasped_mask = np.isin(masks, self.grasped_obj_ids)
            points = points[~grasped_mask]
            colors = colors[~grasped_mask]
            masks = masks[~grasped_mask]

        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        points = np.asarray(pcd_downsampled.points)
        colors = np.asarray(pcd_downsampled.colors).astype(np.uint8)

        return points, colors

    def reset(self):
        obs = self.env.reset()
        self.init_obs = obs
        self.latest_obs = obs
        return obs

    def apply_action(self, action):
        """
        Applies an action in the environment and updates the state.

        Args:
            action: The action to apply.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        assert self.task is not None, "Please load a task first"
        action = self._process_action(action)
        obs, reward, terminate = self.task.step(action)
        obs = self._process_obs(obs)
        self.latest_obs = obs
        self.latest_reward = reward
        self.latest_terminate = terminate
        self.latest_action = action
        self._update_visualizer()
        grasped_objects = self.rlbench_env._scene.robot.gripper.get_grasped_objects()
        if len(grasped_objects) > 0:
            self.grasped_obj_ids = [obj.get_handle() for obj in grasped_objects]
        return obs, reward, terminate

    def move_to_pose(self, pose, speed=None):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            speed: The speed at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([pose, [self.latest_action[-1]]])
        return self.apply_action(action)
    
    def open_gripper(self):
        """
        Opens the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [1.0]])
        return self.apply_action(action)

    def close_gripper(self):
        """
        Closes the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [0.0]])
        return self.apply_action(action)

    def set_gripper_state(self, gripper_state):
        """
        Sets the state of the gripper.

        Args:
            gripper_state: The target state for the gripper.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [gripper_state]])
        return self.apply_action(action)

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([self.init_obs.gripper_pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([self.init_obs.gripper_pose, [self.latest_action[-1]]])
        return self.apply_action(action)

    def get_ee_pose(self):
        assert self.latest_obs is not None, "Please reset the environment first"
        return np.concatenate([self.latest_obs['robot0_eef_pos'], self.latest_obs['robot0_eef_quat']])

    def get_ee_pos(self):
        return self.latest_obs['robot0_eef_pos']

    def get_ee_quat(self):
        return self.latest_obs['robot0_eef_quat']

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return self.init_obs.gripper_open

    def _reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """
        self.init_obs = None
        self.latest_obs = None
        self.latest_reward = None
        self.latest_terminate = None
        self.latest_action = None
        self.grasped_obj_ids = None
        # scene-specific helper variables
        self.arm_mask_ids = None
        self.gripper_mask_ids = None
        self.robot_mask_ids = None
        self.obj_mask_ids = None
        self.name2ids = {}  # first_generation name -> list of ids of the tree
        self.id2name = {}  # any node id -> first_generation name
   
    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(ignore_robot=False, ignore_grasped_obj=False)
            self.visualizer.update_scene_points(points, colors)
    
    def _process_obs(self, obs):
        """
        Processes the observations, specifically converts quaternion format from xyzw to wxyz.

        Args:
            obs: The observation to process.

        Returns:
            The processed observation.
        """
        quat_xyzw = obs.gripper_pose[3:]
        quat_wxyz = np.concatenate([quat_xyzw[-1:], quat_xyzw[:-1]])
        obs.gripper_pose[3:] = quat_wxyz
        return obs

    def _process_action(self, action):
        """
        Processes the action, specifically converts quaternion format from wxyz to xyzw.

        Args:
            action: The action to process.

        Returns:
            The processed action.
        """
        quat_wxyz = action[3:7]
        quat_xyzw = np.concatenate([quat_wxyz[1:], quat_wxyz[:1]])
        action[3:7] = quat_xyzw
        return action