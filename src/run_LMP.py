from utils.arguments import get_config
from modules.interfaces import setup_LMP
from utils.visualizers import ValueMapVisualizer
from envs.robocasa_env import VoxPoserRobocasa
from utils.utils import set_lmp_objects
import numpy as np
import argparse

def run_LMP(args):
    config = get_config(config_path='src/configs/robocasa_config.yaml')
    visualizer = ValueMapVisualizer(config['visualizer'])
    env = VoxPoserRobocasa(visualizer=visualizer, 
                            task_name=args.task_name, 
                            task_config=config['task'])
    lmps, lmp_env = setup_LMP(env, config, debug=False)
    voxposer_ui = lmps['plan_ui']
    obs = env.reset()
    set_lmp_objects(lmps, env.get_visible_object_names()) 
    voxposer_ui(env.env.get_ep_meta()['lang'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LMP with VoxPoser on RoboCasa environment')
    parser.add_argument('--task_name', type=str, default='ArrangeVegetables')
    args = parser.parse_args()
    run_LMP(args)