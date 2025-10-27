import os
import numpy as np
import open3d as o3d
import json
import robosuite as suite
from robosuite import load_composite_controller_config
from robocasa.utils.env_utils import create_env
from utils import normalize_vector, bcolors

TABLE_ALIAS =["table","gripper0", 'right', "cutting", "light", "window", 'counter', "floor", 'cab', 'stack', 'wall', 'mobilebase0', 'utensil',"robot0"]

class VoxPoserRobocasa:
    def __init__(self, env_name="PnPCounterToCab", camera_names=["robot0_agentview_center", "robot0_eye_in_hand"], robot="Panda"):

        # 컨트롤러 설정
        self.controller_config = load_composite_controller_config(
            controller="BASIC",
            robot=robot,
        )

        # RoboCasa 환경 생성
        self.env = create_env(
            env_name=env_name,
            render_onscreen=False,
            camera_names=camera_names,
            camera_depths=True
        )
        # 카메라 이름 저장 및 lookat 벡터 초기화
        self.camera_names = list(camera_names)
        self._init_lookat_vectors()
        self.workspace_bounds_min = np.array([-0.8, -0.8, 0.8])
        self.workspace_bounds_max = np.array([0.8, 0.8, 1.6])
        
        # 객체 이름 매핑 로드
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robocasa_object_names.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.task_object_names = json.load(f)
        self.current_env_name = env_name
        self._reset_task_variables()
        self.map_size = 128
        
        print(f"✅ VoxPoser RoboCasa 환경 초기화 완료: {env_name} with {robot}")

    def _init_lookat_vectors(self):
        """각 카메라의 월드 좌표계 기준 시선 벡터를 초기화합니다.

        우선 MuJoCo의 카메라 회전 행렬(`cam_mat`)을 사용해 카메라 프레임의 +Z 축을
        월드 프레임으로 변환합니다. 실패 시 보수적으로 [0, 0, -1]을 기본값으로 사용합니다.
        """
        self.lookat_vectors = {}
        forward_vector_cam = np.array([0.0, 0.0, 1.0])
        for cam_name in getattr(self, 'camera_names', []) or []:
            lookat_world = None
            try:
                # robosuite/mujoco: cam_mat shape (ncam, 9)
                cam_id = self.env.sim.model.camera_name2id(cam_name)
                R_flat = self.env.sim.model.cam_mat[cam_id]
                R = np.array(R_flat).reshape(3, 3)
                lookat_world = R @ forward_vector_cam
            except Exception:
                # 안전한 기본값: 카메라가 씬을 바라본다고 가정해 -Z를 기본값으로 사용
                lookat_world = np.array([0.0, 0.0, -1.0])
            self.lookat_vectors[cam_name] = normalize_vector(lookat_world)
    
    def get_visible_object_names(self):
        visible_objects = []
        for cam in self.env.camera_names:
            cam_id = self.env.sim.model.camera_name2id(cam)
            seg = self.env.sim.render(camera_name=cam, height=self.map_size, width=self.map_size, segmentation=True)[:,:,1]
            visible_geom_ids = np.unique(seg)
            visible_objs = set([self.env.sim.model.body_id2name(self.env.sim.model.geom_bodyid[int(gid)]).split("_")[0] for gid in visible_geom_ids if gid >= 0 and self.env.sim.model.geom_id2name(int(gid)) is not None])
            visible_objects.extend(visible_objs)
        visible_objects = list(set(visible_objects))
        visible_objects = [obj for obj in visible_objects if obj not in TABLE_ALIAS]
        print("All visible objects:", visible_objects)
        return visible_objects
    
    def load_task(self, task_name):
        print(f"🔄 작업 로드 중: {task_name}")
        self.current_env_name = task_name
        self._reset_task_variables()
        
        # 환경 재생성이 필요한 경우
        if hasattr(self.env, 'close'):
            self.env.close()
        
        # 새 환경 생성
        self.env = create_env(
            env_name=task_name,
            robots="Panda",
            controller_configs=self.controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="robot0_eye_in_hand",
            use_camera_obs=True,
            use_object_obs=True,
            control_freq=20,
            horizon=1000,
            reward_shaping=True,
            camera_heights=self.map_size,
            camera_widths=self.map_size,
            camera_depths=True
        )
        # 환경 카메라 이름을 최신 상태로 반영하고 lookat 벡터 갱신
        try:
            self.camera_names = list(self.env.camera_names)
        except Exception:
            # env가 camera_names 속성을 제공하지 않는 경우 기존 값을 유지
            pass
        self._init_lookat_vectors()
        # 객체 ID 매핑 업데이트
        self._update_object_mappings()

    def fetch_obj_segmentation(self, cam_name, query_name):
        seg = self.env.sim.render(camera_name=cam_name, height=self.map_size, width=self.map_size, segmentation=True)[:,:,1]
        geom_ids = [self.env.sim.model.geom_name2id(geom_name) \
                                for geom_name in self.env.sim.model.geom_names \
                                if query_name in geom_name]
        obj_geom_mask = np.isin(seg, geom_ids)
        # obj_body_mask = np.isin(seg[..., 1], body_id)
        # Debug
        # from PIL import Image
        # rgb_img = self.env.sim.render(camera_name=cam_name, height=256, width=256)
        # Image.fromarray(rgb_img).save("rgb_img.png")
        # Image.fromarray(obj_geom_mask).save("obj_geom_mask.png")
        return obj_geom_mask

    def get_3d_obs_by_name(self, query_name):
        """
        객체 이름으로 3D 점군 관찰 및 법선 벡터 추출
        
        Args:
            query_name (str): 쿼리할 객체 이름
            
        Returns:
            tuple: (객체 점들, 객체 법선들)
        """
        # self._update_object_mappings()
        
        # 모든 카메라에서 점군과 마스크 수집
        points, masks, normals = [], [], []
        
        for cam in self.env.camera_names:
            if f"{cam}_depth" in self.latest_obs:
                # 깊이 이미지에서 점군 생성
                depth_img = self.latest_obs[f"{cam}_depth"].squeeze()
                seg = self.fetch_obj_segmentation(cam, query_name)
                rgb_img = self.latest_obs[f"{cam}_image"]
                
                # 카메라 내재 매개변수 (가정값)
                fx = fy = self.map_size  # 이미지 크기와 동일하게 가정
                cx = cy = self.map_size / 2  # 이미지 중심
                
                # 점군 생성
                h, w = depth_img.shape
                y, x = np.mgrid[0:h, 0:w]
                
                # 유효한 깊이 값만 사용
                valid_mask = depth_img > 0 & seg
                z = depth_img[valid_mask]
                y_valid = y[valid_mask]
                x_valid = x[valid_mask]
                
                # 3D 점 계산
                x_3d = (x_valid - cx) * z / fx
                y_3d = (y_valid - cy) * z / fy
                z_3d = z
                
                cam_points = np.stack([x_3d, y_3d, z_3d], axis=1)
                points.append(cam_points)
                
                # 간단한 법선 벡터 추정
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cam_points)
                pcd.estimate_normals()
                cam_normals = np.asarray(pcd.normals)
                
                # 시선 벡터를 사용해 법선 벡터 조정
                flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
                cam_normals[flip_indices] *= -1
                normals.append(cam_normals)
                
                # 마스크는 모든 점을 유효한 것으로 가정
                masks.append(np.ones(len(cam_points), dtype=int))
        
        if not points:
            # 점군이 없으면 기본값 반환
            default_pos = np.array([[0.3, -0.4, 0.9]])
            default_normals = np.array([[0.0, 0.0, 1.0]])
            return default_pos, default_normals
        
        points = np.concatenate(points, axis=0)
        normals = np.concatenate(normals, axis=0)
        
        # 복셀 다운샘플링
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.01)
        
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)
        
        return obj_points, obj_normals
    
    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        전체 장면의 3D 점군 관찰 및 색상 추출
        
        Args:
            ignore_robot (bool): 로봇 점들 무시 여부
            ignore_grasped_obj (bool): 잡은 객체 점들 무시 여부
            
        Returns:
            tuple: (장면 점들, 색상들)
        """
        points, colors = [], []
        
        for cam in self.camera_names:
            if f"{cam}_depth" in self.latest_obs:
                depth_img = self.latest_obs[f"{cam}_depth"]
                rgb_img = self.latest_obs[f"{cam}_image"]
                
                # 카메라 내재 매개변수
                fx = fy = self.map_size
                cx = cy = self.map_size / 2
                
                # 점군 생성
                try:
                    depth_img = depth_img.squeeze()
                except:
                    breakpoint()
                h, w = depth_img.shape
                y, x = np.mgrid[0:h, 0:w]
                
                valid_mask = depth_img > 0
                z = depth_img[valid_mask]
                y_valid = y[valid_mask]
                x_valid = x[valid_mask]
                
                x_3d = (x_valid - cx) * z / fx
                y_3d = (y_valid - cy) * z / fy
                z_3d = z
                
                cam_points = np.stack([x_3d, y_3d, z_3d], axis=1)
                cam_colors = rgb_img[valid_mask]
                
                points.append(cam_points)
                colors.append(cam_colors)
        
        if not points:
            return np.array([]), np.array([])
        
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        
        # 작업 공간 내의 점들만 유지
        in_bounds = (
            (points[:, 0] >= self.workspace_bounds_min[0]) & 
            (points[:, 0] <= self.workspace_bounds_max[0]) &
            (points[:, 1] >= self.workspace_bounds_min[1]) & 
            (points[:, 1] <= self.workspace_bounds_max[1]) &
            (points[:, 2] >= self.workspace_bounds_min[2]) & 
            (points[:, 2] <= self.workspace_bounds_max[2])
        )
        
        points = points[in_bounds]
        colors = colors[in_bounds]
        
        # 복셀 다운샘플링
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.01)
        
        points = np.asarray(pcd_downsampled.points)
        colors = (np.asarray(pcd_downsampled.colors) * 255).astype(np.uint8)
        
        return points, colors
    
    def reset(self):
        """
        환경 및 작업 초기화
        
        Returns:
            tuple: (작업 설명, 초기 관찰)
        """
        obs = self.env.reset()
        self.init_obs = obs
        self.latest_obs = obs
        
        # 작업 설명 생성
        descriptions = [f"Complete the {self.current_env_name} task"]
        
        return descriptions, obs
    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([self.init_obs['robo0_eef_pos'], [1.0]]) # default gripper open
        else:
            action = np.concatenate([self.init_obs['robot0_eef_pos'], [self.latest_action[-1]]])
        return self.apply_action(action)

    def apply_action(self, action):
        """
        환경에서 액션 적용 및 상태 업데이트
        
        Args:
            action: 적용할 액션
            
        Returns:
            tuple: (최신 관찰, 보상, 종료 플래그)
        """
        # TODO(jshan): add navigation action (zero for now)
        print("[WARNING] We are now applying navigation action as zero for now")
        action = np.concatenate((np.zeros(4),np.array(action)))
        obs, reward, done, info = self.env.step(action)
        self.latest_obs = obs
        self.latest_reward = reward
        self.latest_terminate = done
        self.latest_action = action
        
        return obs, reward, done
    
    def move_to_pose(self, pose, speed=None):
        """
        로봇 팔을 특정 포즈로 이동
        
        Args:
            pose: 목표 포즈
            speed: 이동 속도 (현재 구현되지 않음)
            
        Returns:
            tuple: (최신 관찰, 보상, 종료 플래그)
        """
        # OSC_POSE 컨트롤러용 액션 생성
        action = np.zeros(7)
        action[:3] = pose[:3] - self.get_ee_pos()  # 위치 차이
        action[3:6] = pose[3:6]  # 회전 (오일러 각도)
        action[6] = self.get_last_gripper_action()  # 그리퍼 상태 유지
        
        return self.apply_action(action)
    
    def open_gripper(self):
        """그리퍼 열기"""
        action = np.zeros(7)
        action[6] = -1.0  # 그리퍼 열기
        return self.apply_action(action)
    
    def close_gripper(self):
        """그리퍼 닫기"""
        action = np.zeros(7)
        action[6] = 1.0  # 그리퍼 닫기
        return self.apply_action(action)
    
    def set_gripper_state(self, gripper_state):
        """
        그리퍼 상태 설정
        
        Args:
            gripper_state: 목표 그리퍼 상태
            
        Returns:
            tuple: (최신 관찰, 보상, 종료 플래그)
        """
        action = np.zeros(7)
        action[6] = gripper_state
        return self.apply_action(action)
    
    def get_ee_pose(self):
        """엔드이펙터 포즈 반환"""
        if self.latest_obs is None:
            return np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        
        # RoboCasa의 로봇 상태에서 엔드이펙터 위치 추출
        eef_pos_key = None
        eef_quat_key = None
        
        for key in self.latest_obs.keys():
            if "eef_pos" in key.lower():
                eef_pos_key = key
            elif "eef_quat" in key.lower():
                eef_quat_key = key
        
        if eef_pos_key and eef_quat_key:
            pos = self.latest_obs[eef_pos_key]
            quat = self.latest_obs[eef_quat_key]
            return np.concatenate([pos, quat])
        else:
            # 기본값 반환
            return np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    
    def get_ee_pos(self):
        """엔드이펙터 위치 반환"""
        return self.get_ee_pose()[:3]
    
    def get_ee_quat(self):
        """엔드이펙터 쿼터니언 반환"""
        return self.get_ee_pose()[3:]
    
    def get_last_gripper_action(self):
        """마지막 그리퍼 액션 반환"""
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return -1.0  # 기본적으로 열린 상태
    
    def _reset_task_variables(self):
        """작업 관련 변수 초기화"""
        self.init_obs = None
        self.latest_obs = None
        self.latest_reward = None
        self.latest_terminate = None
        self.latest_action = None
        self.grasped_obj_ids = None
        self.name2ids = {}
        self.id2name = {}
    
    def _update_object_mappings(self):
        """객체 ID 매핑 업데이트"""
        # RoboCasa에서는 객체 매핑이 다르게 작동할 수 있음
        # 환경별로 적절한 매핑 구현 필요
        if self.current_env_name in self.task_object_names:
            name_mapping = self.task_object_names[self.current_env_name]
            for exposed_name, internal_name in name_mapping:
                # 임시로 ID 1로 설정 (실제로는 환경에서 추출해야 함)
                self.name2ids[exposed_name] = [1]
                self.id2name[1] = exposed_name
    