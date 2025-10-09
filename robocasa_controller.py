#!/usr/bin/env python3
import sys
import os
from datetime import datetime
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'src'))
sys.path.append('/workspace/policy/Voxposer/src')
import numpy as np
import time
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.robocasa_env import VoxPoserRobocasa
import cv2
import argparse
from utils import set_lmp_objects
import imageio


class RoboCasaController:
    """
    RoboCasa 환경에서 VoxPoser를 사용한 문 열기 컨트롤러
    """
    
    def __init__(self, env_name="OpenSingleDoor", visualize=True, debug=False):
        print(f"   환경: {env_name}")
        print(f"   시각화: {visualize}")
        print(f"   디버그: {debug}")
        
        # 설정 로드
        self.config = get_config(env='robocasa')
        self.debug = debug
        
        # 시각화 설정
        if visualize:
            self.visualizer = ValueMapVisualizer(self.config['visualizer'])
        else:
            self.visualizer = None
        
        # RoboCasa 환경 생성
        self.env = VoxPoserRobocasa(
            env_name=env_name,
            camera_names=["robot0_agentview_center", "robot0_eye_in_hand"],
            robot="Panda"
        )
        
        # VoxPoser LMP 설정
        print("🧠 VoxPoser LMP 설정 중...")
        self.lmps, self.lmp_env = setup_LMP(self.env, self.config, debug=debug)
        self.voxposer_ui = self.lmps['plan_ui']
        
        # 작업 상태 변수
        self.current_task = None
        self.task_completed = False
        self.step_count = 0
        self.max_steps = 200
    
    def load_task(self):
        # 환경 초기화
        descriptions, obs = self.env.reset()
        available_objects = self.env.get_visible_object_names()
        # LMP 객체 설정
        set_lmp_objects(self.lmps, available_objects)
        return obs

    def execute_task(self, task_instruction="Open the door"):
        # 작업 로드
        obs = self.load_task()
        
        # 비디오 기록기 초기화 (간결 버전)
        def select_image_key(d):
            for k in ("robot0_robotview_image", "robot0_eye_in_hand_image"):
                if k in d:
                    return k
            for k in d.keys():
                if k.endswith("_image"):
                    return k
            return None

        def prep_rgb(frame):
            if frame is None:
                return None
            if frame.dtype != np.uint8:
                mx = float(np.max(frame)) if frame.size else 1.0
                frame = (frame * 255.0).clip(0, 255).astype(np.uint8) if mx <= 1.0 else np.clip(frame, 0, 255).astype(np.uint8)
            if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
                norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                frame = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
            return frame

        video_writer, video_path = None, None
        try:
            img_key0 = "robot0_agentview_center_image"
            if img_key0 is not None:
                f0 = prep_rgb(obs[img_key0])
                if f0 is not None:
                    h, w = f0.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    vdir = os.path.join(os.getcwd(), "videos")
                    os.makedirs(vdir, exist_ok=True)
                    video_path = os.path.join(vdir, f"robocasa_task_{ts}.mp4")
                    video_writer = []
        except Exception as _e:
            if self.debug:
                print(f"⚠️ 비디오 기록 초기화 실패: {_e}")
        
        # 시작 시간 기록
        start_time = time.time()
        
        try:
            # VoxPoser를 사용하여 작업 계획 및 실행
            print("🗺️ VoxPoser로 작업 계획 생성 중...")
            
            # LMP를 통한 작업 계획
            plan_result = self.voxposer_ui((task_instruction, obs))
            # _obs, _, _ = self.env.apply_action(np.zeros(7))
            
            if self.debug:
                print(f"📋 계획 결과: {plan_result}")
            
            # VoxPoser 계획 결과를 사용한 작업 실행
            print("🤖 VoxPoser 계획 기반 작업 실행 시작...")
            self.step_count = 0
            
            # VoxPoser가 생성한 실행 정보가 있는지 확인
            if plan_result is not None:
                print(f"📋 VoxPoser 계획 결과 타입: {type(plan_result)}")
                if self.debug:
                    print(f"📋 계획 결과 내용: {plan_result}")
                
                # VoxPoser 실행 정보에서 궤적 추출 시도
                execution_info = None
                if hasattr(plan_result, '__iter__') and len(plan_result) > 0:
                    execution_info = plan_result[0] if isinstance(plan_result, list) else plan_result
                elif hasattr(plan_result, 'controller_infos'):
                    execution_info = plan_result
                print(f"execution_info: {execution_info}")
                if execution_info and 'controller_infos' in execution_info:
                    # VoxPoser가 생성한 컨트롤러 정보 사용
                    controller_infos = execution_info['controller_infos']
                    print(f"📋 VoxPoser가 생성한 {len(controller_infos)}개 웨이포인트 실행")
                    
                    for waypoint_idx, controller_info in controller_infos.items():
                        if self.step_count >= self.max_steps:
                            break
                            
                        # 현재 상태 확인
                        current_ee_pos = self.env.get_ee_pos()
                        
                        if self.debug and self.step_count % 10 == 0:
                            print(f"Step {self.step_count}: EE Position = {current_ee_pos}")
                        
                        # VoxPoser가 생성한 액션 실행
                        try:
                            if 'action' in controller_info:
                                action = controller_info['action']
                                _obs, _, _ = self.env.apply_action(action)
                            else:
                                # 기본 액션 (no-op)
                                _obs, _, _ = self.env.apply_action(np.zeros(12))
                        except Exception as _e:
                            if self.debug:
                                print(f"⚠️ 웨이포인트 {waypoint_idx} 실행 실패: {_e}")

                        # 프레임 기록
                        video_writer.append(_obs['robot0_agentview_center_image'][::-1][::-1])
                        time.sleep(0.05)  # 웨이포인트 간 대기
                        self.step_count += 1
                        
                        # 간단한 완료 조건 (목표에 도달했는지 확인)
                        if hasattr(controller_info, 'mp_info') and controller_info['mp_info'] == 0:
                            print("🎯 목표 웨이포인트 도달!")
                            break
                else:
                    print("⚠️ VoxPoser 실행 정보에서 컨트롤러 정보를 찾을 수 없음")
            else:
                print("⚠️ VoxPoser 계획 결과가 비어있음")
                
            # 기본 실행 모드 (VoxPoser 결과가 없거나 실행 완료 후)
            if self.step_count == 0:
                print("🔄 기본 실행 모드로 전환...")
                while self.step_count < self.max_steps and not self.task_completed:
                    # 현재 상태 확인
                    current_ee_pos = self.env.get_ee_pos()
                    
                    if self.debug and self.step_count % 50 == 0:
                        print(f"Step {self.step_count}: EE Position = {current_ee_pos}")
                    
                    # 환경을 한 스텝 진행 (no-op 액션으로 관찰 업데이트)
                    try:
                        _obs, _, _ = self.env.apply_action(np.random.randn(12))
                    except Exception as _e:
                        print(f"⚠️ 스텝 진행 실패: {_e}")
                        
                    video_writer.append(_obs['robot0_agentview_center_image'][::-1])
                    time.sleep(0.01)  # 시각화를 위한 짧은 대기
                    self.step_count += 1
            
            
        except Exception as e:
            print(f"❌ 작업 실행 중 오류 발생: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
        imageio.mimsave(video_path, video_writer, fps=10.0)

        """작업 결과 출력"""
        print("\n" + "="*60)
        print("="*60)
        print(f"⏱️  총 실행 시간: {total_time:.2f}초")
        print(f"📊 총 스텝 수: {self.step_count}")
        print(f"🏆 작업 성공 여부: {'성공' if self.task_completed else '미완료'}")
        print(f"⚡ 평균 스텝 시간: {total_time/self.step_count:.4f}초")
        print("="*60)
    
    def run_demo(self, instructions=None):    
        instruction = self.env.env.get_ep_meta()['lang']
        print(f"\n{'='*50}")
        print(f"{instruction} task")
        print('='*50)
        self.execute_task(instruction)    

def main():
    # 컨트롤러 생성
    controller = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="OpenDrawer")
    args = parser.parse_args()
    controller = RoboCasaController(
        env_name=args.env_name,
        debug=True
    )
    controller.run_demo()
    

if __name__ == "__main__":
    main()
