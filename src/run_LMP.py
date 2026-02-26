import sys
import traceback

sys.path.append("/home/jisu/workspace/safety/robotics-safety/benchmark/robocasa/robocasa")

from utils.arguments import get_config
from utils.visualizers import ValueMapVisualizer
from utils.utils import set_lmp_objects
from modules.interfaces import setup_LMP
from envs.robocasa_env import VoxPoserRobocasa

TASK_TYPE = "navigation"

NAVIGATE_TASKS = [
    "NavigateKitchen",
    "NavigateKitchenWithCat",
    "NavigateKitchenWithDog",
    "NavigateKitchenWithMug",
    "NavigateKitchenWithKettlebell",
    "NavigateKitchenWithTowel",
]


def run_single_task(task_name, config):
    print(f"\n{'='*60}")
    print(f"  Running task: {task_name}")
    print(f"{'='*60}\n")
    visualizer = ValueMapVisualizer(config['visualizer'])
    env = VoxPoserRobocasa(visualizer=visualizer,
                            task_name=task_name,
                            task_config=config['task'])
    lmps, lmp_env = setup_LMP(env, config, debug=False)
    voxposer_ui = lmps['plan_ui']
    env.load_task()
    set_lmp_objects(lmps, env.get_visible_object_names())
    voxposer_ui(env.env.get_ep_meta()['lang'])


def run_all_tasks():
    config = get_config(config_path='src/configs/robocasa_config.yaml', task_type=TASK_TYPE)
    results = {}
    for task_name in NAVIGATE_TASKS:
        try:
            run_single_task(task_name, config)
            results[task_name] = "SUCCESS"
        except Exception as e:
            print(f"\n[ERROR] Task {task_name} failed: {e}")
            traceback.print_exc()
            results[task_name] = f"FAILED: {e}"

    print(f"\n{'='*60}")
    print("  Results Summary")
    print(f"{'='*60}")
    for task_name, status in results.items():
        print(f"  {task_name:40s} {status}")


if __name__ == "__main__":
    run_all_tasks()