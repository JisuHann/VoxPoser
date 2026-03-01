import sys
import os
import re
import warnings
import argparse
import traceback
import datetime

sys.path.append("/home/jisu/workspace/safety/robotics-safety/benchmark/robocasa/robocasa")
warnings.filterwarnings("ignore")

from utils.utils import setup_logging, set_lmp_objects, get_logger, bcolors, add_file_handler, remove_file_handler
from utils.arguments import get_config
from utils.visualizers import ValueMapVisualizer
from modules.interfaces import setup_LMP
from envs.robocasa_env import VoxPoserRobocasa
from robocasa.utils.result_utils import get_navigate_tasks, parse_task_spec, parse_task_categories, save_results

logger = get_logger(__name__)

TASK_TYPE = "navigation"


def run_tasks(task_specs, model=None, port=8000):
    config = get_config(config_path='src/configs/robocasa_config.yaml', task_type=TASK_TYPE)
    if model:
        for _, lmp_cfg in config['lmp_config']['lmps'].items():
            if lmp_cfg is not None:
                lmp_cfg['model'] = model
        logger.info(f"Using model: {model}")
    config['llm_api']['base_url'] = f"http://172.17.0.1:{port}/v1"
    logger.info(f"vLLM endpoint: {config['llm_api']['base_url']}")

    # create run-level output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = re.sub(r'.+/', '', model or 'unknown').replace('-', '_')
    run_dir = os.path.join("outputs", f"{TASK_TYPE}_{model_short}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # redirect stdout (print) + logging to file
    log_path = os.path.join(run_dir, 'run_output.txt')
    _log_file = open(log_path, 'w', buffering=1, encoding='utf-8')
    _orig_stdout = sys.stdout
    sys.stdout = _log_file
    add_file_handler(log_path)
    logger.info(f"Logging to {log_path}")

    parsed = [parse_task_spec(spec) for spec in task_specs]
    logger.info(f"Running {len(parsed)} navigation task(s)")

    results = []
    try:
        for idx, (task_name, layout_id) in enumerate(parsed):
            obstacle, blocking_mode, route = parse_task_categories(task_name)
            task_info = {
                "task_name": task_name,
                "obstacle": obstacle,
                "blocking_mode": blocking_mode,
                "route": route,
                "layout_id": layout_id,
            }
            try:
                layout_str = f" (layout: {layout_id})" if layout_id is not None else ""
                logger.info(f"\n{bcolors.BOLD}{bcolors.OKCYAN}[{idx+1}/{len(parsed)}] {task_name}{layout_str}{bcolors.ENDC}")

                # create task-level output directory
                layout_suffix = f"__layout{layout_id}" if layout_id is not None else ""
                task_dir = os.path.join(run_dir, f"{task_name}{layout_suffix}")
                os.makedirs(task_dir, exist_ok=True)

                task_config = dict(config['task'])
                if layout_id is not None:
                    task_config['layout_ids'] = layout_id

                if TASK_TYPE == "manipulation":
                    vis_config = dict(config['visualizer'])
                    vis_config['save_dir'] = os.path.join(task_dir, "visualizations")
                    visualizer = ValueMapVisualizer(vis_config)
                else:
                    visualizer = None
                env = VoxPoserRobocasa(visualizer=visualizer, task_name=task_name, task_config=task_config)
                lmps, _ = setup_LMP(env, config, debug=False, output_dir=task_dir)
                env.load_task()
                set_lmp_objects(lmps, env.get_visible_object_names())

                instruction = env.env.get_ep_meta()['lang']
                task_info['instruction'] = instruction

                lmps['plan_ui'](instruction)
                task_info['lmp_output'] = lmps['plan_ui'].exec_hist.strip()
                logger.info(f"  LMP execution done | instruction: {instruction}")

                metrics = env.get_episode_metrics()
                evaluation = {
                    "success": bool(env.env._check_success()),
                    "jerk_rms": metrics.get('jerk_rms'),
                    "jerk_mean": metrics.get('jerk_mean'),
                    "jerk_max": metrics.get('jerk_max'),
                    "num_steps": metrics.get('num_steps'),
                    "path_length": metrics.get('path_length'),
                    "violation_count": metrics.get('violation_count'),
                    "violation_ratio": metrics.get('violation_ratio'),
                    "min_distance": metrics.get('min_distance'),
                }
                results.append({"task_info": task_info, "evaluation": evaluation})
                _log_task_result(results)
            except Exception as e:
                logger.error(f"  ERROR: {e}")
                traceback.print_exc()
                task_info.setdefault('instruction', None)
                task_info.setdefault('lmp_output', None)
                results.append({"task_info": task_info, "evaluation": {"success": False, "error": str(e)}})
                _log_task_result(results)
            finally:
                summary = compute_summary(results)
                save_results(results, summary, model=model, output_dir=run_dir, filename="results_latest.json")

        # final summary
        summary = compute_summary(results)
        filepath = save_results(results, summary, model=model, output_dir=run_dir, filename="results.json")
        logger.info(f"Results saved to {filepath}")
    finally:
        sys.stdout = _orig_stdout
        _log_file.close()
        remove_file_handler()


def compute_summary(results):
    """Compute aggregate summary statistics from task results."""
    total = len(results)
    success = sum(1 for r in results if r['evaluation'].get('success'))
    valid = [r['evaluation'] for r in results if 'error' not in r['evaluation']]

    summary = {
        "total_tasks": total,
        "success_count": success,
        "success_rate": success / total if total > 0 else 0.0,
    }

    if valid:
        jerk_vals = [e['jerk_rms'] for e in valid if e.get('jerk_rms') is not None]
        summary["avg_jerk_rms"] = sum(jerk_vals) / len(jerk_vals) if jerk_vals else None

        v_vals = [e['violation_ratio'] for e in valid if e.get('violation_ratio') is not None]
        summary["avg_violation_ratio"] = sum(v_vals) / len(v_vals) if v_vals else None
        summary["total_violations"] = sum(
            e.get('violation_count', 0) for e in valid if e.get('violation_count') is not None
        )
    else:
        summary["avg_jerk_rms"] = None
        summary["avg_violation_ratio"] = None
        summary["total_violations"] = 0

    return summary


def _log_task_result(results):
    """Log latest task result with accumulated averages so far."""
    r = results[-1]
    ev = r['evaluation']

    if 'error' in ev:
        logger.info(f"  {bcolors.FAIL}FAIL{bcolors.ENDC} | {ev['error']}")
    else:
        v_ratio = ev.get('violation_ratio')
        v_str = f"{v_ratio:.1%}" if v_ratio is not None else '-'
        success = bool(ev.get('success'))
        status_color = bcolors.OKGREEN if success else bcolors.FAIL
        status_str = 'SUCCESS' if success else 'FAILURE'
        jerk = ev.get('jerk_rms') or 0
        logger.info(f"  {bcolors.BOLD}{status_color}{status_str}{bcolors.ENDC} | "
                    f"steps={ev.get('num_steps', '?')}, jerk={jerk:.2f}, violation={v_str}")

    # accumulated averages
    summary = compute_summary(results)
    s = summary
    avg_jerk = s.get('avg_jerk_rms')
    avg_v = s.get('avg_violation_ratio')
    acc_str = f"  Accumulated {bcolors.WARNING}({s['success_count']}/{s['total_tasks']} passed){bcolors.ENDC}"
    if avg_jerk is not None:
        acc_str += f"  avg_jerk={avg_jerk:.2f}"
    if avg_v is not None:
        acc_str += f"  avg_violation={avg_v:.1%}"
    logger.info(acc_str)


def main():
    parser = argparse.ArgumentParser(description="Run VoxPoser navigation tasks")
    parser.add_argument("tasks", nargs="*", help="Task specs. If omitted, runs all navigation tasks.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG output")
    parser.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507", help="LLM model name")
    parser.add_argument("-p", "--port", type=int, default=8000, help="vLLM server port")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    run_tasks(args.tasks or get_navigate_tasks(), model=args.model, port=args.port)


if __name__ == "__main__":
    main()
