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

## Quick start

```shell
# 1. start a vLLM server on the host
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8003 --max-model-len 16384

# 2. inside the robocasa docker container, run an eval
python3 src/run_LMP.py \
    -m Qwen/Qwen3-4B-Instruct-2507 -p 8003 \
    --system-prompt default --few-shot default \
    --layout-ids 0 --style-ids 3
```

LMP-only mode (no physics rollout — useful for prompt ablation) is
enabled with `--lmp-only`.

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
