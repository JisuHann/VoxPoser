## VoxPoser implemented in Robocasa

## Setup Instructions

## Running LMP

'''Shell
python -m vllm.entrypoints.openai.api_server \
 --model Qwen/Qwen3-4B \
 --port 8000 \
 --gpu-memory-utilization 0.7 \
 --max-model-len 4096 \
 --trust-remote-code
'''

'''Shell
pytho3n src/run_LMP.py
'''

## Running Demo

## Code Structure

Core to VoxPoser:

- **`playground.ipynb`**: Playground for VoxPoser.
- **`LMP.py`**: Implementation of Language Model Programs (LMPs) that recursively generates code to decompose instructions and compose value maps for each sub-task.
- **`interfaces.py`**: Interface that provides necessary APIs for language models (i.e., LMPs) to operate in voxel space and to invoke motion planner.
- **`planners.py`**: Implementation of a greedy planner that plans a trajectory (represented as a series of waypoints) for an entity/movable given a value map.
- **`controllers.py`**: Given a waypoint for an entity/movable, the controller applies (a series of) robot actions to achieve the waypoint.
- **`dynamics_models.py`**: Environment dynamics model for the case where entity/movable is an object or object part. This is used in `controllers.py` to perform MPC.
- **`prompts/rlbench`**: Prompts used by the different Language Model Programs (LMPs) in VoxPoser.

Environment and utilities:

- **`envs`**:
  - **`rlbench_env.py`**: Wrapper of RLBench env to expose useful functions for VoxPoser.
  - **`task_object_names.json`**: Mapping of object names exposed to VoxPoser and their corresponding scene object names for each individual task.
- **`configs/rlbench_config.yaml`**: Config file for all the involved modules in RLBench environment.
- **`arguments.py`**: Argument parser for the config file.
- **`LLM_cache.py`**: Caching of language model outputs that writes to disk to save cost and time.
- **`utils.py`**: Utility functions.
- **`visualizers.py`**: A Plotly-based visualizer for value maps and planned trajectories.
