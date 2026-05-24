## VoxPoser implemented in RoboCasa (navigation-only)

Mobile-robot safety-aware navigation built on top of RoboCasa (kitchen sim)
and the VoxPoser LMP (Language Model Program) framework. The original
VoxPoser supported manipulation; this fork strips the arm/gripper path
and focuses entirely on navigating a holonomic base safely through a
populated kitchen.

## End-to-end pipeline

```
                  ┌─────────────────────────────────────────────────────┐
                  │              user instruction                       │
                  │  e.g. "navigate safely to the human while avoiding  │
                  │        obstacles"                                   │
                  └────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          PLANNER LMP                                     │
│  prompts/<env>/planner_prompt.txt                                        │
│                                                                          │
│  Input  : instruction + scene 'objects = [...]' list                     │
│  Output : one or more composer("...") calls                              │
│           composer("move to the goal while keeping at least 60cm from    │
│                     the obstacle and slowing down when near it")         │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         COMPOSER LMP                                     │
│  prompts/<env>/composer_prompt.txt                                       │
│                                                                          │
│  Input  : a natural-language composer() string                           │
│  Output : python code that invokes the three map LMPs and finally calls  │
│           execute_navigation(movable, affordance_map, avoidance_map,     │
│                              velocity_map, rotation_map)                 │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       MAP LMPs (four of them)                            │
│                                                                          │
│  get_affordance_map  →  cells that pull the robot in (goal region)       │
│      e.g. "a point 25cm in front of the goal"                            │
│           → set_pixel_by_radius(map, pt, radius_cm=25, value=1)          │
│  get_avoidance_map   →  cells the robot must keep clearance from         │
│      e.g. "60cm from the obstacle"                                       │
│           → set_pixel_by_radius(map, obstacle, radius_cm=60, value=1)    │
│  get_velocity_map    →  per-cell desired speed (1.0 = nominal)           │
│      e.g. "slow to 30% velocity within 60cm of the obstacle"             │
│           → set_pixel_by_radius(map, obstacle, radius_cm=60, value=0.3)  │
│  get_rotation_map    →  per-cell desired yaw (NaN = "keep current")      │
│      e.g. "face the goal" → yaw = atan2(goal - pos), stamped on cells    │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                COSTMAP FUSION  (modules/planners.py)                     │
│                                                                          │
│  Fuses affordance and avoidance into one (H × W) costmap:                │
│    target_field   = distance-transform of affordance (low near goal)     │
│    obstacle_field = gaussian-blurred avoidance (soft repulsion halo)     │
│    costmap        = α · target_field + β · obstacle_field                │
│                                                                          │
│  α, β live in configs/robocasa_config.yaml under `planner:`              │
│  (target_map_weight, obstacle_map_weight). Larger β = wider berth        │
│  around obstacles, possibly at the cost of path length.                  │
│                                                                          │
│  Velocity and rotation maps do NOT enter the costmap — they are          │
│  applied later as per-waypoint annotations (see below).                  │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│             PathPlanner.navigation_optimize (modules/planners.py)        │
│                                                                          │
│  Primary search: A* (use_astar: true in config).                         │
│    - 8-connected grid, edge cost = costmap value of target cell          │
│    - heuristic = Euclidean distance to nearest affordance cell           │
│    - optional dilation of the obstacle mask by `robot_radius_cells`      │
│      so the path keeps a body-sized clearance; retries with smaller      │
│      inflation if no full path is found, picking the run that reaches    │
│      the goal (or gets closest)                                          │
│                                                                          │
│  Fallback: greedy descent if A* finds no path even at zero inflation     │
│  (rare; emits a warning).                                                │
│                                                                          │
│  Post-process: smooth the raw pixel path; greedy paths are additionally  │
│  clipped by max_curvature so the controller can interpolate smoothly.    │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│         WAYPOINT ANNOTATION  (modules/interfaces.py:_path2traj_*)        │
│                                                                          │
│  For each waypoint along the pixel path, look up:                        │
│    velocity = velocity_map[row, col]   →  scaled motor speed (1 = nominal)│
│    yaw      = rotation_map[row, col]   →  desired heading, or NaN to     │
│                                            keep the current yaw          │
│                                                                          │
│  This is how slowdowns near hazards reach the controller: the velocity   │
│  map is built earlier by get_velocity_map (e.g. "slow to 30% within      │
│  60cm of the obstacle"), and the planner's path inherits that scalar     │
│  cell-by-cell. The controller passes (x, y, yaw, velocity) to the env.   │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              NavigationController (modules/controllers.py)               │
│                                                                          │
│  For each waypoint: send (x, y, yaw, velocity) to RoboCasa env via       │
│  env.apply_navigation_action(...).                                       │
│  Replans every N waypoints with the latest scene observation.            │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  RoboCasa kitchen sim (envs/robocasa_env.py)             │
│                                                                          │
│  MuJoCo physics. Returns RGB cameras, point cloud, robot pose,           │
│  obstacle poses, violation events. SafeNavigate scenes include posed     │
│  humans, pets, fragile dishware, and fixed kitchen fixtures.             │
└──────────────────────────────────────────────────────────────────────────┘
```

The four map LMPs and the composer LMP share a system prompt — that
prompt can be the plain `default_system_prompt.txt` or the
`safety_aware_system_prompt.txt` which instructs the model to scale
clearance and speed by inferred obstacle character.

## Pure-pursuit navigator (current default)

Since 2026-05-24 the default `NAV_MODE` is **`pure_pursuit`** with
**replan + 4-direction escape**. A lookahead target slides
continuously along the path, the projection cursor only moves forward
(no U-shape teleports), and a workspace-bounds guard aborts the
episode if the controller diverges. When stuck, a donut wedge marker
(core 0.15 m free, outer 0.5 m blocked) triggers a replan; if that
also stalls, the robot tries 4 escape directions and picks the
first-improvement one. All knobs default to the best-config values
from the 2026-05-24 sweep — `env` overrides exist but you don't have
to set anything to get the recommended setup.

| env var | default | meaning |
|---|---|---|
| `NAV_MODE` | `pure_pursuit` | navigator (legacy: `holonomic`, `translate_only`, `holo_seq`, `holo_cont`) |
| `PP_LOOKAHEAD_M` | `0.5` | lookahead distance (m) along the projected path |
| `PP_REPLAN_MAX` | `4` | max replan attempts before aborting |
| `PP_GOAL_TOL` | `0.25` | goal-reached threshold (m), controller-side |
| `PP_STUCK_GOAL_R` | `0.6` | "stuck near goal" radius (m) — treat as arrived |
| `PP_DECEL_R` | `0.4` | start decelerating within this radius of the goal (m) |
| `PP_TURN_SLOW` | `100` | turn-priority slowdown gain (high = strong slowdown when yawing) |
| `PP_ESCAPE_ENABLED` | `1` | toggle 4-direction first-improvement escape |
| `PP_OMEGA_MAX` | `0.1` | max angular velocity (rad/step) |
| `START_CLEAR_CELLS` | `2` | start-clear disk radius (cells) — keeps real counters in the map |
| `ROBOT_FOOTPRINT_Z_MAX` | `0.6` | robot footprint height cap (m) |

LMP cache and Voxposer outputs:

```
src/cache/                                    # disk LMP result cache (delete to invalidate)
outputs/<run_name>/                           # one dir per evaluation run
├── layout{0..9}/
│   └── NavigateKitchen<obstacle><mode>Route<X>/
│       ├── run.log                           # full per-task log (SUCCESS/FAILURE line at end)
│       ├── task_info.json                    # per-task summary (success, viol, lmp_output)
│       ├── trajectory_log.json               # robot trajectory, obstacle distance history
│       ├── voxposer_overview.png             # final map composition + path
│       ├── initial_topview.png               # scene snapshot
│       ├── voxposer_dump.npz                 # raw maps + planner output
│       └── *_image.mp4                       # per-camera videos
├── results.json                              # aggregated summary (last worker to exit writes)
└── setup_w*.log                              # per-worker setup log
```

## Quick start

```shell
# 1. start a vLLM server on the host (Qwen3-14B example, port 8000, TP=2)
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B \
  --served-model-name Qwen/Qwen3-14B \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --trust-remote-code

# 2. inside the robocasa docker container, run an eval — no PPENV needed,
#    defaults are already the best config (LH=0.5, replan=4, escape on, …)
docker exec -it robocasa-eval-0 bash
cd /workspace/policy/Voxposer
MUJOCO_GL=egl python3 src/run_LMP.py \
    -m Qwen/Qwen3-14B -p 8000 -w 0 \
    -o outputs/smoke \
    --system-prompt default --few-shot default \
    --layout-ids 0 --style-ids 3 \
    NavigateKitchenCatBlockingRouteA
```

A small 4-B Qwen3 also works (TP=1 is enough):

```shell
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8003 --max-model-len 16384
docker exec robocasa-eval-0 bash -c "
  cd /workspace/policy/Voxposer && MUJOCO_GL=egl \
  python3 src/run_LMP.py -m Qwen/Qwen3-4B-Instruct-2507 -p 8003 -w 0 \
    --system-prompt default --few-shot default \
    --layout-ids 0 --style-ids 3
"
```

LMP-only mode (no physics rollout — useful for prompt ablation) is
enabled with `--lmp-only`.

## Multi-container parallel evaluation (6 docker workers)

Each container has a dedicated MuJoCo framebuffer, so one task at a
time per container. With six `robocasa-eval-0..5` containers, you can
run 6 workers in parallel against a single host vLLM:

```shell
# tasks for a single layout (write 1 task per line, 150 lines for a full sweep)
cat > /tmp/task_list.txt <<'EOF'
NavigateKitchenCatBlockingRouteA
NavigateKitchenCatBlockingRouteB
# ... 150 tasks total
EOF
TASKS=$(cat /tmp/task_list.txt | tr '\n' ' ')

for idx in 0 1 2 3 4 5; do
  docker exec -d robocasa-eval-$idx bash -c "
    cd /workspace/policy/Voxposer && OMP_NUM_THREADS=8 MUJOCO_GL=egl \
    python3 src/run_LMP.py -m Qwen/Qwen3-14B -p 8000 -w $idx \
      -o outputs/sweep --system-prompt default --few-shot default \
      --layout-ids $idx --style-ids 3 $TASKS \
      > /tmp/sweep_d${idx}.log 2>&1
  "
done
```

A 5-layout × 6-shard wave script (used for the 690-task prompt
ablation in `robotics-safety/scripts/_pp_bench_690.sh`) is the
recommended template if you want a sequential layout-by-layout sweep
that fully utilises all six workers.

## CLI knobs that change behaviour

- `--system-prompt {default, safety_aware}` — picks
  `prompts/robocasa_navigation_system/<sp>_system_prompt.txt`. The
  `safety_aware` overlay instructs the model to scale clearance and
  speed by inferred obstacle character.
- `--few-shot {default, safety_aware, safety_aware_v2}` — picks the
  example directory under `prompts/`. `safety_aware_v2` adds the
  CRITICAL goal-vs-obstacle distinction note for tasks where the
  destination is itself a human.
- `--lmp-only` — skip the physics rollout and only capture LMP outputs.
- `--layout-ids` — kitchen layouts (0..9, minus the three banned ones).
- `--style-ids` — kitchen decoration styles (0..11). **All current
  experiments fix this to `3`** (consistent visuals so style does not
  confound the prompt/model comparison). Do not vary unless you are
  specifically studying style sensitivity.

## Code layout

```
src/
├── run_LMP.py                  # entry point, argparse, per-task runner
├── core/LMP.py                 # one LMP = (prompt → vLLM → code → exec)
├── modules/
│   ├── interfaces.py           # LMP_interface: APIs the LLM sees
│   │                           #   (parse_query_obj, set_pixel_by_radius,
│   │                           #    get_*_map, execute_navigation)
│   ├── planners.py             # PathPlanner.navigation_optimize
│   └── controllers.py          # NavigationController.execute
├── envs/robocasa_env.py        # MuJoCo wrapper + safety metrics
├── configs/robocasa_config.yaml
├── utils/                      # logging, prompts, cache, ssi
└── prompts/
    ├── robocasa_navigation/                    (baseline few-shot)
    ├── robocasa_navigation_safety_aware/       (character-aware few-shot)
    ├── robocasa_navigation_safety_aware_v2/    (safety_aware + CRITICAL note)
    ├── robocasa_navigation_system/             (system prompt variants)
    └── robocasa_navigation_tier/               (legacy tier-implicit)
```
