import os

from core.LMP import LMP
from utils.utils import get_clock_time, normalize_vector, pointat2quat, TermColors, Observation, VoxelIndexingWrapper, get_logger, DynamicObservation
from utils.visualization import save_image, save_array, visualize_voxel, save_map_to_image, save_video_images
import numpy as np
from modules.planners import PathPlanner
import time
from scipy.ndimage import distance_transform_edt
import transforms3d
from modules.controllers import NavigationController
from envs.robocasa_env import VoxPoserRobocasa, KITCHEN_GROUP_SUBNAMES
from tqdm import tqdm
from transforms3d.euler import quat2euler

logger = get_logger(__name__)

def _vec2quat(*args):
    """Convert a direction vector to a quaternion (yaw rotation in the xy-plane).
    Accepts either vec2quat(v) or vec2quat(ai, aj, ak) (euler-like, for LLM compatibility)."""
    if len(args) == 3:
        return transforms3d.euler.euler2quat(args[0], args[1], args[2])
    v = np.asarray(args[0], dtype=float)
    yaw = np.arctan2(v[1], v[0]) if np.linalg.norm(v[:2]) > 1e-8 else 0.0
    return transforms3d.euler.euler2quat(yaw, 0, 0)

YAW_THRESHOLD_DEFAULT = 0.35
DIST_THRESHOLD_DEFAULT = 0.05
EE_ALIAS = ['ee', 'endeffector', 'end_effector', 'end effector', 'gripper', 'hand']
TABLE_ALIAS = ['table', 'desk', 'workstation', 'work_station', 'work station', 'workspace', 'work_space', 'work space']

class NavigationLMPInterface():

  def __init__(self, env, lmp_config, controller_config, planner_config, env_name='rlbench', nav_controller_config=None, output_dir=None):
    self._env = env
    self._env_name = env_name
    self._cfg = lmp_config
    self._map_size = self._cfg['map_size']  # legacy scalar (manipulation cube)
    self._output_dir = output_dir or "."
    # Navigation grid is rectangular (map_h × map_w) with isotropic cells of
    # `resolution_cm` size. Falls back to map_size square if env doesn't
    # expose map_h/map_w (e.g. manipulation env).
    # IMPORTANT: must compute _map_h/_map_w BEFORE instantiating PathPlanner.
    # The planner's `_postprocess_path` clips path coords to [0, map_size-1].
    # If passed legacy scalar 100, navigation paths with rows > 99 (L0/L1/L3
    # have map_h=122/142/162) get clamped to 99 — verified bug: L3 path_pixel[0]
    # row=99 instead of row=139, causing 2m start gap and robot to drive
    # away from its actual position.
    _nav_h = int(getattr(self._env, 'map_h', self._map_size))
    _nav_w = int(getattr(self._env, 'map_w', self._map_size))
    self._planner = PathPlanner(planner_config, map_size=max(_nav_h, _nav_w, self._map_size))
    self._nav_controller = NavigationController(self._env, controller_config)
    if nav_controller_config is None:
        nav_controller_config = {}
    self._yaw_threshold = nav_controller_config.get('yaw_threshold', YAW_THRESHOLD_DEFAULT)
    self._dist_threshold = nav_controller_config.get('dist_threshold', DIST_THRESHOLD_DEFAULT)
    self._max_steps_per_waypoint = nav_controller_config.get('max_steps_per_waypoint', 100)
    # Navigation mode: holonomic (default, body-frame v_x/v_y/omega) vs
    # translate_only (rotate-first toward lookahead wp, then forward-only).
    # NAV_MODE env-var overrides config.
    # Default = pure_pursuit (best config per sweep 2026-05-24, replaces
    # legacy holonomic). NAV_MODE env-var or config override.
    self._nav_mode = os.environ.get('NAV_MODE', nav_controller_config.get('mode', 'pure_pursuit'))
    assert self._nav_mode in ('holonomic', 'translate_only', 'holo_seq', 'holo_cont', 'pure_pursuit'), f"unknown nav_mode: {self._nav_mode}"
    # When True, parse_query_obj() / detect() auto-resolve fixture queries
    # (coffee_machine, sink, stove, ...) to fixture.pos instead of the
    # visible point-cloud centroid. Default True — flip to False to
    # restore legacy centroid behaviour.
    self._use_fixture_pos = bool(self._cfg.get('use_fixture_pos', True))
    # detect() memoization — within one task, static-scene objects are queried
    # many times (74+ calls/task in v3 prompts). Cache by lowercase name so
    # repeated parse_query_obj('human') / detect('sink') reuse the same
    # occupancy_map + point-cloud computation. Robot/mobile_base/gripper are
    # NEVER cached since the robot moves during the task.
    self._detect_cache = {}
    self._detect_skip_names = ('mobile_base', 'robot_mobile_base', 'gripper', 'robot0', 'ee')

    self._map_h = _nav_h
    self._map_w = _nav_w

    # calculate size of each voxel (resolution). Use rectangular dims so x/y
    # resolutions are equal (both = resolution_cm).
    _ws_extent = self._env.workspace_bounds_max - self._env.workspace_bounds_min
    self._resolution = np.array([
        _ws_extent[0] / self._map_w,
        _ws_extent[1] / self._map_h,
        _ws_extent[2] / self._map_size,
    ])

  # ======================================================
  # == functions exposed to LLM
  # ======================================================
  def get_ee_pos(self):
    return self._world_to_voxel(self._env.get_ee_pos())
  
  def detect(self, obj_name):
    """Return an observation dict for the object.

    When the env config has `interfaces.use_fixture_pos: true` (default),
    queries that match a known kitchen *fixture* (coffee_machine, sink,
    stove, fridge, microwave, oven, dishwasher) automatically resolve
    `position` / `_position_world` to the fixture's reference pose
    (`fixture.pos`) instead of the visible point-cloud centroid.

    Why this matters: visible-mesh centroid can be 30cm–1m off
    fixture.pos (fridge sliding doors, sink basin interior, coffee
    machine front face). Goal-success uses fixture.pos with a 0.5m
    threshold, so anchoring affordance to fixture.pos closes that gap.

    Movable obstacles (cat, dog, human, ...) are not in the fixtures
    registry so they always fall back to visible centroid — which is
    what we want (avoidance halo around the actual mesh).
    """
    print("object name:", obj_name)
    # Memoize: static-scene objects (sink, fridge, human, cat, etc.) don't
    # move during a task — return cached Observation. Skip cache for robot
    # (moves) and EE alias (depends on arm config).
    _obj_lc = (obj_name or '').lower().strip()
    _is_cacheable = (
        _obj_lc not in self._detect_skip_names
        and _obj_lc not in EE_ALIAS
    )
    if _is_cacheable and _obj_lc in self._detect_cache:
        return self._detect_cache[_obj_lc]
    # if obj_name.lower() == 'robot_mobile_base':
    #   import sys, os, logging
    #   # sys.stdout is redirected to log file by run_LMP.py
    #   # restore it to terminal for pdb interaction
    #   logging.disable(logging.CRITICAL)
    #   _saved_stdout = sys.stdout
    #   sys.stdout = sys.__stdout__
    #   sys.stderr = sys.__stderr__
    #   print("\n" + "="*60)
    #   print("PDB_BREAKPOINT_HIT: detect('robot_mobile_base')")
    #   print("="*60)
    #   sys.stdout.flush()
    #   breakpoint()
    #   # restore after continuing
    #   sys.stdout = _saved_stdout
    #   sys.stderr = sys.__stderr__
    #   logging.disable(logging.NOTSET)
    # Grouped 'kitchen' label → union of all kitchen-furniture point clouds
    # so the LLM can avoid the generic kitchen as a single semantic obstacle.
    if obj_name.lower() == 'kitchen':
      obs_dict = dict()
      pcs, normals = [], []
      for _sub in KITCHEN_GROUP_SUBNAMES:
        try:
          (_, _), (sub_pc, sub_n) = self._env.get_3d_obs_by_name(_sub)
          if sub_pc is not None and len(sub_pc):
            pcs.append(np.asarray(sub_pc))
            normals.append(np.asarray(sub_n))
        except Exception:
          continue
      if not pcs:
        # Fall back to default detect path so we don't crash
        return self.detect('counter')
      obj_pc = np.concatenate(pcs, axis=0)
      obj_normal = np.concatenate(normals, axis=0)
      voxel_map = self._points_to_voxel_map(obj_pc)
      aabb_min = self._world_to_voxel(np.min(obj_pc, axis=0))
      aabb_max = self._world_to_voxel(np.max(obj_pc, axis=0))
      obs_dict['occupancy_map'] = voxel_map
      obs_dict['name'] = 'kitchen'
      obs_dict['position'] = self._world_to_pos_coords(np.mean(obj_pc, axis=0))
      obs_dict['aabb'] = np.array([aabb_min, aabb_max])
      obs_dict['_position_world'] = np.mean(obj_pc, axis=0)
      obs_dict['_point_cloud_world'] = obj_pc
      obs_dict['normal'] = normalize_vector(obj_normal.mean(axis=0))
      _result = Observation(obs_dict)
      if _is_cacheable:
        self._detect_cache[_obj_lc] = _result
      return _result
    if obj_name.lower() in EE_ALIAS:
      obs_dict = dict()
      obs_dict['name'] = obj_name
      obs_dict['position'] = self.get_ee_pos()
      obs_dict['aabb'] = np.array([self.get_ee_pos(), self.get_ee_pos()])
      obs_dict['_position_world'] = self._env.get_ee_pos()
    elif obj_name.lower() in TABLE_ALIAS:
      offset_percentage = 0.1
      x_min = self._env.workspace_bounds_min[0] + offset_percentage * (self._env.workspace_bounds_max[0] - self._env.workspace_bounds_min[0])
      x_max = self._env.workspace_bounds_max[0] - offset_percentage * (self._env.workspace_bounds_max[0] - self._env.workspace_bounds_min[0])
      y_min = self._env.workspace_bounds_min[1] + offset_percentage * (self._env.workspace_bounds_max[1] - self._env.workspace_bounds_min[1])
      y_max = self._env.workspace_bounds_max[1] - offset_percentage * (self._env.workspace_bounds_max[1] - self._env.workspace_bounds_min[1])
      table_max_world = np.array([x_max, y_max, 0])
      table_min_world = np.array([x_min, y_min, 0])
      table_center = (table_max_world + table_min_world) / 2
      obs_dict = dict()
      obs_dict['name'] = obj_name
      obs_dict['position'] = self._world_to_pos_coords(table_center)
      obs_dict['_position_world'] = table_center
      obs_dict['normal'] = np.array([0, 0, 1])
      obs_dict['aabb'] = np.array([self._world_to_pos_coords(table_min_world), self._world_to_pos_coords(table_max_world)])
    else:
      obs_dict = dict()
      (workspace_pc, _), (obj_pc, obj_normal) = self._env.get_3d_obs_by_name(obj_name)
      voxel_map = self._points_to_voxel_map(obj_pc)
      aabb_min = self._world_to_voxel(np.min(obj_pc, axis=0))
      aabb_max = self._world_to_voxel(np.max(obj_pc, axis=0))
      obs_dict['occupancy_map'] = voxel_map  # in voxel frame
      obs_dict['name'] = obj_name
      # Default: mesh point-cloud centroid for position
      _pos_world = np.mean(obj_pc, axis=0)
      # SPECIAL CASE — mobile_base / robot_mobile_base: use the actual
      # base anchor (mobilebase0_base body_xpos) instead of mesh centroid.
      # The mesh centroid includes the pedestal/arm and shifts up to 30cm
      # from the planar base anchor depending on robot pose. Planner
      # start_pos uses this `position`; if it disagrees with the
      # controller's `robot0_base_pos`, the planner produces a path
      # starting at the wrong cell — robot then has to pre-traverse a
      # 0.05–0.30m gap before the planner's first wp is "reached", and
      # the visualised START marker drifts off the actual robot.
      if obj_name.lower() in ('mobile_base', 'robot_mobile_base', 'mobilebase0'):
        try:
          _sim = self._env.env.sim
          _bid = _sim.model.body_name2id('mobilebase0_base')
          _bp = np.asarray(_sim.data.body_xpos[_bid])[:3].copy()
          # Keep mesh centroid z (the base body is at z=0 but planner
          # treats x,y only; safe to overwrite all 3 since downstream
          # uses xy.
          _pos_world = _bp
          logger.debug(f"detect('{obj_name}'): using base anchor "
                       f"({_bp.tolist()}) instead of mesh centroid "
                       f"({np.mean(obj_pc, axis=0).tolist()})")
        except Exception as _e:
          logger.warning(f"detect('{obj_name}'): base-anchor lookup failed "
                         f"({_e}); falling back to mesh centroid")
      obs_dict['position'] = self._world_to_pos_coords(_pos_world)
      obs_dict['aabb'] = np.array([aabb_min, aabb_max])  # in voxel frame
      obs_dict['_position_world'] = _pos_world           # in world frame
      obs_dict['_point_cloud_world'] = obj_pc            # in world frame
      obs_dict['normal'] = normalize_vector(obj_normal.mean(axis=0))
    # Auto-resolve to fixture.pos when configured and a matching kitchen
    # fixture exists. Movable obstacles fall through (no fixture match).
    if getattr(self, '_use_fixture_pos', True):
      _fpos = self._lookup_fixture_pos(obj_name)
      if _fpos is not None:
        obs_dict['_position_world'] = np.asarray(_fpos)
        obs_dict['position'] = self._world_to_pos_coords(np.asarray(_fpos))
    # #78: expose fixture orientation for LMP face-toward intent.
    # `.normal` is camera-derived per-point average → biased to -z, useless
    # for "face the fixture" rotation. fixture.rot is the sim-side mounted
    # yaw (z-axis, radians). fixture_front = rot + π/2 matches env's
    # `compute_robot_base_placement_pose` convention (kitchen.py:692) so
    # robot facing fixture_front is properly aligned to use the fixture.
    _frot = self._lookup_fixture_rot(obj_name)
    if _frot is not None:
      obs_dict['fixture_yaw']   = float(_frot)
      obs_dict['fixture_front'] = float(_frot) + float(np.pi / 2)
    # A3: override with env.target_pos when this object IS the navigation
    # target fixture. fixture.pos = mesh centroid (e.g. sink_island_group at
    # counter middle), but env.target_pos = "robot's required approach pose"
    # (computed by compute_robot_base_placement_pose — counter edge minus
    # robot offset). For sink-like fixtures these differ by ~0.5m. Without
    # this override, LMP affordance lands at fixture.pos and robot stops far
    # from the actual goal_pos used for success eval. Verified L1 v32 RouteB:
    # sink at (3.75, -2.30) vs goal at (3.75, -1.80) → 0.50m semantic gap.
    try:
      _kitchen = getattr(self._env, "env", None)
      if _kitchen is not None:
        _tf = getattr(_kitchen, "target_fixture", None)
        _tp = getattr(_kitchen, "target_pos", None)
        if _tf is not None and _tp is not None:
          _tf_name = (getattr(_tf, "name", "") or "").lower()
          _q = obj_name.lower().replace(" ", "_")
          # Same alias logic as _lookup_fixture_pos to match canonical names.
          _alias = {
              "coffee_machine": ("coffee", "coffeemachine"),
              "sink":           ("sink",),
              "stove":          ("stove", "stovetop"),
              "stovetop":       ("stove", "stovetop"),
              "fridge":         ("fridge",),
              "microwave":      ("microwave", "micro"),
              "oven":           ("oven",),
              "dishwasher":     ("dishwasher",),
          }.get(_q, (_q,))
          if any(k in _tf_name for k in _alias):
            obs_dict['_position_world'] = np.asarray(_tp)
            obs_dict['position'] = self._world_to_pos_coords(np.asarray(_tp))
            logger.debug(f"detect('{obj_name}'): override _position_world to env.target_pos {np.asarray(_tp).tolist()} (was fixture.pos)")
    except Exception as _e:
      logger.debug(f"target_pos override failed for '{obj_name}': {_e}")
    object_obs = Observation(obs_dict)
    if _is_cacheable:
      self._detect_cache[_obj_lc] = object_obs
    return object_obs

  def _lookup_fixture_pos(self, obj_name):
    """Return (x, y, z) of the matching kitchen fixture's reference frame.
    Returns None if no fixture matches `obj_name`. Used by `detect(.., as_target=True)`
    to anchor affordance/goal to fixture.pos instead of visible centroid."""
    try:
      kitchen = getattr(self._env, "env", None)
      if kitchen is None: return None
      fixtures = getattr(kitchen, "fixtures", None) or {}
      target_alias = {
          "coffee_machine": ("coffee", "coffeemachine"),
          "sink":           ("sink",),
          "stove":          ("stove", "stovetop"),
          "stovetop":       ("stove", "stovetop"),
          "fridge":         ("fridge",),
          "microwave":      ("microwave", "micro"),
          "oven":           ("oven",),
          "dishwasher":     ("dishwasher",),
      }
      keys = target_alias.get(obj_name.lower(), (obj_name.lower(),))
      for fname, fix in fixtures.items():
        fl = fname.lower()
        if any(k in fl for k in keys) and hasattr(fix, "pos"):
          return np.asarray(fix.pos)
    except Exception as e:
      logger.debug(f"_lookup_fixture_pos({obj_name}) failed: {e}")
    return None

  def _lookup_fixture_rot(self, obj_name):
    """Return fixture.rot (z-axis yaw, radians) for matched kitchen fixture.
    Returns None if no fixture matches. Used by detect() to expose
    fixture_yaw / fixture_front so LMP can encode "face the fixture" intent
    without relying on the noisy camera-derived `.normal`."""
    try:
      kitchen = getattr(self._env, "env", None)
      if kitchen is None: return None
      fixtures = getattr(kitchen, "fixtures", None) or {}
      target_alias = {
          "coffee_machine": ("coffee", "coffeemachine"),
          "sink":           ("sink",),
          "stove":          ("stove", "stovetop"),
          "stovetop":       ("stove", "stovetop"),
          "fridge":         ("fridge",),
          "microwave":      ("microwave", "micro"),
          "oven":           ("oven",),
          "dishwasher":     ("dishwasher",),
      }
      keys = target_alias.get(obj_name.lower(), (obj_name.lower(),))
      for fname, fix in fixtures.items():
        fl = fname.lower()
        if any(k in fl for k in keys) and hasattr(fix, "rot"):
          return float(fix.rot)
    except Exception as e:
      logger.debug(f"_lookup_fixture_rot({obj_name}) failed: {e}")
    return None

  def save_image(self, array, save_path="tmp.png"):
      save_image(array, save_path)

  def save_array(self, array, save_name="tmp.npy"):
      save_array(array, save_name)

  def visualize_voxel(self, voxel_maps, voxel_size=0.1):
      visualize_voxel(voxel_maps, voxel_size)

  def execute_navigation(self, movable_obs_func, affordance_map=None, avoidance_map=None, rotation_map=None,
              velocity_map=None, **kwargs):
    """
    Plan a navigation path then follow it with the controller.

    Side effect for downstream visualisation:
      Saves `{output_dir}/voxposer_dump.npz` containing a per-plan-iter record
      of (path_pixel, traj_world, affordance/avoidance/rotation/velocity maps,
      start_pos). Used by `scripts/visualize_voxposer_task.py`.

    Args:
      movable_obs_func: callable returning observation of the body to be moved
      affordance_map: callable returning 2D target pixel map
      avoidance_map: callable returning 2D obstacle pixel map
      rotation_map: callable returning 2D rotation map
      velocity_map: callable returning 2D velocity map
    """
    # VLM-generated code may pass an already-evaluated Observation instead of a callable.
    # Wrap non-callable observations so the rest of the pipeline works.
    if not callable(movable_obs_func):
      obs = movable_obs_func
      movable_obs_func = lambda: obs
    if rotation_map is None:
      rotation_map = self._get_default_voxel_map('rotation', task='navigation')
    if velocity_map is None:
      velocity_map = self._get_default_voxel_map('velocity', task='navigation')
    if avoidance_map is None:
      avoidance_map = self._get_default_voxel_map('obstacle', task='navigation')
    object_centric = False
    execute_info = []
    controller_infos = dict()
    if affordance_map is not None:
      for plan_iter in range(self._cfg['max_plan_iter']):
        step_info = dict()
        movable_obs = movable_obs_func()
        # LMP-generated get_*_map code can either: (a) return None (omitted
        # `ret_val =`) or (b) raise — e.g. kimi-vl sometimes calls cm2index
        # with a scalar direction for vague queries ("any of them"). Both
        # paths fall back to the default voxel map so the episode survives
        # one LLM-quality glitch rather than RETRYABLE-ing 3× and giving up.
        def _safe_map(fn, kind):
            try:
                v = fn()
                if v is None:
                    import logging
                    logging.getLogger(__name__).warning(
                        f'[execute_navigation] {kind}_map returned None → default'
                    )
                    return self._get_default_voxel_map(kind, task='navigation')()
                return v
            except Exception as _e:
                import logging
                logging.getLogger(__name__).warning(
                    f'[execute_navigation] {kind}_map raised {type(_e).__name__}: {_e} → default'
                )
                return self._get_default_voxel_map(kind, task='navigation')()
        _affordance_map = _safe_map(affordance_map, 'target')
        _avoidance_map = _safe_map(avoidance_map, 'obstacle')
        _rotation_map = _safe_map(rotation_map, 'rotation')
        _velocity_map = _safe_map(velocity_map, 'velocity')
        _avoidance_map = self._preprocess_avoidance_pixel_map(_avoidance_map, _affordance_map, movable_obs)
        start_pos = movable_obs['position'][:2]
        start_time = time.time()
        # Inflation enabled with multi-attempt fallback — A* tries the
        # largest radius first (collision-safe) and progressively reduces
        # if no path found, ending at 0 inflation (point robot guarantee).
        # Baseline radius = mobilebase geom rbound / 2 (taken directly from
        # MuJoCo model; for the wheeled_base rbound ≈ 0.54m → baseline 0.27m).
        # Half-rbound matches the planar footprint better than the full
        # enclosing sphere (which over-counts vertical extent), and keeps the
        # planner aligned with the avoidance-clear footprint that uses real
        # geom geometry. Falls back to baseline/2, 1, 0 cells in tight kitchens.
        try:
          _xy = self._compute_pixel_resolution()
          _cell_m = float(np.asarray(_xy).min())
          # Inflation = base PLANAR radius (R), since A* moves the base CENTER
          # and robot extends R outward. geom_rbound is the 3D enclosing-sphere
          # radius which over-counts vertical extent (tall arms make sphere big
          # but irrelevant for floor footprint).
          # Use geom_aabb on mobile_base geoms to get XY half-extents directly.
          # ROBOT_RADIUS_M env var overrides.
          _override = os.environ.get('ROBOT_RADIUS_M', '').strip()
          if _override:
            _robot_radius_m = float(_override)
          else:
            _robot_radius_m = 0.40   # safe planar default (PandaOmron base ~0.6x0.6 → diag/2 ≈ 0.42)
            try:
              _model = self._env.env.sim.model
              _max_xy = 0.0
              for _gid in range(_model.ngeom):
                _bid = int(_model.geom_bodyid[_gid])
                _bn = (_model.body_id2name(_bid) or '').lower()
                if 'mobilebase' in _bn:
                  # geom_aabb is (cx, cy, cz, hx, hy, hz) in geom-local frame.
                  # For floor footprint we want hypot(hx, hy) — half-diagonal in XY.
                  _aabb = _model.geom_aabb[_gid]
                  _hx = float(_aabb[3]); _hy = float(_aabb[4])
                  _xy_r = float(np.hypot(_hx, _hy))
                  if _xy_r > _max_xy:
                    _max_xy = _xy_r
              if _max_xy > 0:
                _robot_radius_m = _max_xy   # planar half-diagonal
            except Exception:
              pass
          _robot_radius_cells = max(1, int(np.ceil(_robot_radius_m / _cell_m)))
          logger.info(f"[planner inflation] robot_radius_m={_robot_radius_m:.3f} "
                      f"cell_m={_cell_m:.3f} → robot_radius_cells={_robot_radius_cells}")
        except Exception:
          _robot_radius_cells = 0
        path_pixel, planner_info = self._planner.navigation_optimize(start_pos, _affordance_map, _avoidance_map,
                                                                      object_centric=object_centric,
                                                                      robot_radius_cells=_robot_radius_cells)
        logger.debug(f'[{get_clock_time()}] planner time: {time.time() - start_time:.3f}s')
        assert len(path_pixel) > 0, 'path_pixel is empty'
        step_info['path_pixel'] = path_pixel
        step_info['planner_info'] = planner_info
        traj_world = self._path2traj_navigation(path_pixel, _avoidance_map, _rotation_map, _velocity_map)
        traj_world = traj_world[:self._cfg['num_waypoints_per_plan']]
        step_info['start_pos'] = start_pos
        step_info['plan_iter'] = plan_iter
        step_info['movable_obs'] = movable_obs
        step_info['traj_world'] = traj_world
        step_info['affordance_map'] = _affordance_map
        step_info['rotation_map'] = _rotation_map
        step_info['velocity_map'] = _velocity_map
        step_info['avoidance_map'] = _avoidance_map

        if self._cfg.get('visualize', False):
          save_map_to_image(_avoidance_map, path_pixel, save_path=os.path.join(self._output_dir, f"plan_iter_{plan_iter}.png"))

        logger.debug(f'[{get_clock_time()}] executing path ({len(traj_world)} waypoints)')
        controller_infos = dict()
        step_idx = 0

        # Rotation handling — 3 modes via env:
        #  ROTATE_FIRST_ENABLED=1 (default): pre-loop burst rotate in place
        #     toward wp[1]. Fast (~26 steps for 90°) but ~195mm drift.
        #  INCREMENTAL_ROTATE_ENABLED=1: skip rotate-first burst. Inside
        #     outer loop, action[2] omega is the only rotation source —
        #     gradual alignment over many small steps. omega clip 0.05
        #     → 0.14°/step → drift 0.3mm/step accumulates over many steps.
        #     Total drift same as rotate-first (degrees-of-rotation invariant)
        #     BUT spread across translation steps → robot has time to make
        #     progress toward wp while rotating.
        #  Default holonomic without flags = same as rotate-first.
        _incremental_rotate = os.environ.get('INCREMENTAL_ROTATE_ENABLED', '0') == '1'
        # pure_pursuit rotates incrementally inside its own pursuit loop (small
        # per-step omega → sub-mm coupling drift). The legacy rotate-first
        # BURST below spins in place with no translation compensation → drifts
        # ~2.5mm/° (148° turn → ~0.37m), shoving the robot into a wall before
        # the pursuit loop even starts. Never run the burst in pure_pursuit.
        _rotate_first_enabled = (
            not _incremental_rotate
            and os.environ.get('ROTATE_FIRST_ENABLED', '1') == '1'
            and self._nav_mode != 'pure_pursuit'
        )
        if _rotate_first_enabled and len(traj_world) >= 2:
          # rotate-first targets the NEXT SEGMENT direction (wp[1]→wp[2]) so the
          # robot pre-aligns to where it will be heading AFTER the first wp.
          # Falls back to (wp[0]→wp[1]) if only 2 wps in path.
          if len(traj_world) >= 3:
            first_dxy = np.asarray(traj_world[2][0]) - np.asarray(traj_world[1][0])
          else:
            first_dxy = np.asarray(traj_world[1][0]) - np.asarray(traj_world[0][0])
          if np.linalg.norm(first_dxy) > 0.05:
            # Body +x in world = (-cos α, sin α) where α=real_yaw (R(π-yaw)).
            # To make body face direction (dx, dy): α = atan2(dy, -dx).
            first_tangent_yaw = float(np.arctan2(first_dxy[1], -first_dxy[0]))
            _q = self._env.env._get_observations()['robot0_base_quat']
            cur_yaw0 = quat2euler([_q[3], _q[0], _q[1], _q[2]])[2]
            init_delta = (first_tangent_yaw - cur_yaw0 + np.pi) % (2 * np.pi) - np.pi
            ROTATE_FIRST_TRIGGER_RAD = np.deg2rad(60)
            ROTATE_FIRST_TOL_RAD = np.deg2rad(15)
            # No fixed step limit — wait until aligned. Safety cap 500 only
            # to prevent runaway (would mean omega action somehow not rotating).
            ROTATE_FIRST_MAX_STEPS = 500
            if abs(init_delta) > ROTATE_FIRST_TRIGGER_RAD:
              _obs0 = self._env.env._get_observations()
              _start_pos_world = _obs0['robot0_base_pos'][:2].copy()
              logger.info(f'[{get_clock_time()}] rotate-first phase: '
                          f'init delta_yaw={np.degrees(init_delta):+.1f}deg '
                          f'(> {np.degrees(ROTATE_FIRST_TRIGGER_RAD):.0f}deg trigger), '
                          f'start_pos=({_start_pos_world[0]:.3f},{_start_pos_world[1]:.3f})')
              for _rf_step in range(ROTATE_FIRST_MAX_STEPS):
                _obs = self._env.env._get_observations()
                _q = _obs['robot0_base_quat']
                _cy = quat2euler([_q[3], _q[0], _q[1], _q[2]])[2]
                _d = (first_tangent_yaw - _cy + np.pi) % (2 * np.pi) - np.pi
                _cur_pos = _obs['robot0_base_pos'][:2]
                _drift_world = _cur_pos - _start_pos_world
                if abs(_d) <= ROTATE_FIRST_TOL_RAD:
                  logger.info(f'[{get_clock_time()}] rotate-first done in '
                              f'{_rf_step + 1} steps (remaining delta='
                              f'{np.degrees(_d):+.1f}deg, drift='
                              f'{np.linalg.norm(_drift_world)*1000:.1f}mm)')
                  break
                # Pure rotation, no translation compensation. v_x compensation
                # (any direction) is environment-dependent — would push robot
                # into walls in tight starting positions. Instead, accept the
                # drift and correct AFTER rotation via nearest-wp mapping
                # (NEAREST_WP_AFTER_ROTATE). General across all scenes.
                _vx_comp = float(os.environ.get('ROTATE_FIRST_VX_COMP', '0.0'))
                _vy_comp = float(os.environ.get('ROTATE_FIRST_VY_COMP', '0.0'))
                _omega = float(np.clip(_d * 10.0, -1.0, 1.0))
                rotate_action = np.array([_vx_comp, _vy_comp, _omega])
                self._env.apply_navigation_action(rotate_action)
              else:
                logger.warning(f'[{get_clock_time()}] rotate-first hit step '
                               f'limit ({ROTATE_FIRST_MAX_STEPS}); proceeding')

              # === Drift correction after rotate-first ===
              # rotate-first inherently drifts ~2.5mm/° (best case with v_x=-0.5).
              # For 90° rotation: ~225mm displacement. Robot's actual position is
              # no longer at start_pos, so original path (traj_world) is offset.
              # Two strategies (env-var selectable):
              #   REPLAN_AFTER_ROTATE=1 : re-run A* from new pos with cached maps
              #   TRANSLATE_BACK=1      : translate back to start_pos before main loop
              # Default: do nothing (legacy behavior).
              _drift_strategy_replan = os.environ.get('REPLAN_AFTER_ROTATE', '0') == '1'
              _drift_strategy_back   = os.environ.get('TRANSLATE_BACK_AFTER_ROTATE', '0') == '1'
              if (_drift_strategy_replan or _drift_strategy_back):
                _obs_post = self._env.env._get_observations()
                _pos_post = _obs_post['robot0_base_pos'][:2]
                _drift_post = _pos_post - _start_pos_world
                _drift_mag = float(np.linalg.norm(_drift_post))
                logger.info(f'[drift-correction] post-rotate drift={_drift_mag*1000:.1f}mm '
                            f'world=({_drift_post[0]*1000:+.0f},{_drift_post[1]*1000:+.0f})mm')
                if _drift_mag > 0.05:
                  if _drift_strategy_back:
                    # Holonomic translation back to start_pos (body-frame target)
                    _q_post = _obs_post['robot0_base_quat']
                    _yaw_post = quat2euler([_q_post[3], _q_post[0], _q_post[1], _q_post[2]])[2]
                    for _bs in range(200):
                      _obs2 = self._env.env._get_observations()
                      _p2 = _obs2['robot0_base_pos'][:2]
                      _q2_ = _obs2['robot0_base_quat']
                      _yaw2 = quat2euler([_q2_[3], _q2_[0], _q2_[1], _q2_[2]])[2]
                      _dxw = _start_pos_world[0] - _p2[0]
                      _dyw = _start_pos_world[1] - _p2[1]
                      _err = float(np.hypot(_dxw, _dyw))
                      if _err < 0.05:
                        logger.info(f'[drift-correction] translate-back done in {_bs+1} steps')
                        break
                      # Controller's body→world is R(π - yaw), not R(yaw).
                      # Audit confirmed (yaw=0/180° tests showed 180° dir error
                      # with R(yaw) formula). World→body inverse uses R(π-yaw)^T.
                      _vxb = -_dxw*np.cos(_yaw2) + _dyw*np.sin(_yaw2)
                      _vyb = -_dxw*np.sin(_yaw2) - _dyw*np.cos(_yaw2)
                      _kp = 5.0
                      _act = np.array([
                        float(np.clip(_vxb*_kp, -1.0, 1.0)),
                        float(np.clip(_vyb*_kp, -1.0, 1.0)),
                        0.0,
                      ])
                      self._env.apply_navigation_action(_act)
                    else:
                      logger.warning(f'[drift-correction] translate-back hit step limit')
                  elif _drift_strategy_replan:
                    # Re-run A* from current position with cached maps
                    try:
                      _cur_grid = self._world_to_pos_coords(
                          np.array([_pos_post[0], _pos_post[1], 0.0]))
                      _new_path, _ = self._planner.navigation_optimize(
                          np.asarray(_cur_grid)[:2], _affordance_map, _avoidance_map,
                          robot_radius_cells=_robot_radius_cells)
                      if _new_path is not None and len(_new_path) > 1:
                        _new_traj = self._path2traj_navigation(
                            _new_path, _avoidance_map, _rotation_map, _velocity_map)
                        if len(_new_traj) > 1:
                          traj_world = _new_traj
                          logger.info(f'[drift-correction] replanned: {len(traj_world)} wps from new pos')
                    except Exception as _re:
                      logger.warning(f'[drift-correction] replan failed: {_re}')
              # === End drift correction ===

        # Replan support: cache maps + radius for in-loop A* re-run.
        # Enabled via env var REPLAN_EVERY_N_STEPS (0 = disabled, N>0 =
        # rerun A* from cur_pos every N control steps). Keeps drift from
        # accumulating between robot pose and planner path.
        _replan_n = int(os.environ.get('REPLAN_EVERY_N_STEPS', '0'))
        _cached_aff = _affordance_map
        _cached_avoid = _avoidance_map
        _cached_rot_map = _rotation_map
        _cached_vel_map = _velocity_map
        _cached_rr = _robot_radius_cells

        i = 0
        # NEAREST_WP_AFTER_ROTATE=1 : skip wps that are now behind robot
        # (closer to start than to cur_pos). Picks the wp closest to cur_pos
        # as starting index. Robust to ANY drift magnitude — doesn't assume
        # drift direction or magnitude.
        if os.environ.get('NEAREST_WP_AFTER_ROTATE', '0') == '1':
          _obs_nwp = self._env.env._get_observations()
          _cur_xy = _obs_nwp['robot0_base_pos'][:2]
          _dists = [float(np.linalg.norm(np.asarray(_wp[0]) - _cur_xy))
                    for _wp in traj_world]
          if _dists:
            _nearest = int(np.argmin(_dists))
            # Bias toward forward: skip wps that are at/behind cur_pos
            # (only if nearest is very close to cur_pos). Advance to next wp.
            if _dists[_nearest] < 0.10 and _nearest + 1 < len(traj_world):
              _nearest += 1
            i = _nearest
            logger.info(f'[nearest-wp] cur_xy=({_cur_xy[0]:.3f},{_cur_xy[1]:.3f}) '
                        f'→ start at wp[{i}] '
                        f'(dist={_dists[_nearest]*1000:.0f}mm; total {len(traj_world)} wps)')
        pbar = tqdm(total=max(1, len(traj_world)-i), desc='REACHED waypoint')
        # Global step cap — prevents replan-loop oscillation. Cap at the
        # number of waypoints × per-wp limit so normal tasks aren't affected.
        _global_max_steps = self._max_steps_per_waypoint * 2 * max(len(traj_world), 6)
        _global_start_step = step_idx

        # === PURE-PURSUIT mode (NAV_MODE=pure_pursuit) ===
        # Replaces the discrete waypoint-by-waypoint follower. A lookahead
        # target slides CONTINUOUSLY along the path: each step the robot is
        # projected onto the path polyline, and the target is a point
        # PP_LOOKAHEAD_M of arc length ahead. There is no "reach waypoint i"
        # state → no waypoint-stuck, no overshoot reversal (the projection
        # always finds where the robot actually is), and smooth tracking on
        # curved / U-shaped detour paths. A workspace-bounds guard aborts the
        # episode if the controller diverges off the map (the F0 blowup).
        if self._nav_mode == 'pure_pursuit':
          _path_xy = np.array([np.asarray(w[0])[:2] for w in traj_world], dtype=float)
          _final_xy = _path_xy[-1]
          _fy = np.asarray(traj_world[-1][1])
          _final_yaw = float(_fy.item()) if _fy.size == 1 and np.isfinite(_fy.item()) else None
          # Defaults reflect best config from sweep (2026-05-24): LH=0.5,
          # replan_max=4, goal_tol=0.25, escape on, decel_r=0.4, turn_slow=100.
          _Ld      = float(os.environ.get('PP_LOOKAHEAD_M', '0.5'))
          _pp_kp   = float(os.environ.get('PP_KP', '6.0'))
          # #29 REPLAN: when stuck FAR from goal, try replanning from current
          # position before aborting. Max replans capped to avoid infinite loop.
          _pp_replan_count = 0
          _pp_replan_max   = int(os.environ.get('PP_REPLAN_MAX', '4'))
          # incremental rotation: small omega cap → per-step rotation tiny →
          # rotation-translation coupling drift is sub-mm AND continuously
          # corrected by the pursuit loop (no uncorrected rotate-first burst).
          _omega_max = float(os.environ.get('PP_OMEGA_MAX', '0.1'))
          _kp_rot  = float(os.environ.get('CTRL_KP_ROT', '3.0'))
          _succ_thr = float(os.environ.get('PP_GOAL_TOL', '0.25'))
          _ws_min = np.asarray(self._env.workspace_bounds_min[:2], dtype=float)
          _ws_max = np.asarray(self._env.workspace_bounds_max[:2], dtype=float)
          # PP_STEP_CAP_MULT (default 0.5) halves the cap to bound wedge-case
          # runtime. Successful tasks median ~150 step, max ~500 step → 0.5×
          # (cap ~300-700) leaves margin while cutting wedge cap from 1400→700.
          _pp_max_steps = int(self._max_steps_per_waypoint *
                              max(len(traj_world), 6) *
                              float(os.environ.get('PP_STEP_CAP_MULT', '0.5')))
          _pp_start = step_idx
          # Monotonic projection cursor: the closest-segment search is
          # restricted to [_last_k, _last_k + PP_PROJ_WINDOW] and _last_k
          # only increases. A global closest-segment search teleports
          # across U-shaped detour paths (the far side of the U is
          # Euclidean-close to the near side) → the lookahead jumps the
          # robot straight across the obstacle. Forward-windowed search
          # keeps the projection advancing along the path.
          _last_k = 0
          _proj_win = int(os.environ.get('PP_PROJ_WINDOW', '12'))
          # stuck detection: robot blocked by wall/counter (NOT in the
          # obstacle-avoid list, so min_obstacle_distance stays large) →
          # pure-pursuit commands full-forward forever. Detect no-progress
          # and either treat as arrived (near goal) or abort (far).
          _stuck_win = int(os.environ.get('PP_STUCK_WIN', '40'))
          _stuck_eps = float(os.environ.get('PP_STUCK_EPS', '0.04'))
          _stuck_goal_r = float(os.environ.get('PP_STUCK_GOAL_R', '0.6'))
          _pos_hist = []
          # rotate-first: align to the goal heading IN PLACE before travel.
          # During travel omega then stays ~0 → no rotation-translation
          # coupling drift (the big initial turn otherwise smears the
          # body-frame translation, accumulating ~0.5m cross-track error).
          if os.environ.get('PP_ROTATE_FIRST', '0') == '1' and _final_yaw is not None:
            for _rf in range(int(os.environ.get('PP_ROTATE_FIRST_MAX', '500'))):
              _rfo = self._env.env._get_observations()
              _rfq = _rfo['robot0_base_quat']
              _rfy = quat2euler([_rfq[3], _rfq[0], _rfq[1], _rfq[2]])[2]
              _rfd = (_final_yaw - _rfy + np.pi) % (2 * np.pi) - np.pi
              if abs(_rfd) <= self._yaw_threshold:
                logger.info(f'[pure-pursuit] rotate-first done @ step {step_idx} '
                            f'({_rf} steps, rem={np.degrees(_rfd):+.1f}deg)')
                break
              _rfa = np.array([0.0, 0.0,
                               float(np.clip(_rfd * _kp_rot, -_omega_max, _omega_max))])
              _rfci = self._nav_controller.execute(_rfa)
              _rfci['controller_step'] = step_idx
              for _vc in VoxPoserRobocasa.VIDEO_RECORD_CAMERAS:
                _k = f"{_vc}_image"
                if _k in _rfci['mp_info'][0]:
                  _rfci[_k] = _rfci['mp_info'][0][_k][::-1]
              controller_infos[step_idx] = _rfci
              step_idx += 1
          pbar = tqdm(total=_pp_max_steps, desc='pure-pursuit')
          while True:
            _obs = self._env.env._get_observations()
            _cur = np.asarray(_obs['robot0_base_pos'][:2], dtype=float)
            _q = _obs['robot0_base_quat']
            _cyaw = quat2euler([_q[3], _q[0], _q[1], _q[2]])[2]
            # workspace-bounds abort guard (F0 blowup containment)
            if (_cur[0] < _ws_min[0] - 1.0 or _cur[0] > _ws_max[0] + 1.0 or
                _cur[1] < _ws_min[1] - 1.0 or _cur[1] > _ws_max[1] + 1.0):
              logger.warning(f'[pure-pursuit] robot left workspace ({_cur[0]:.2f},'
                             f'{_cur[1]:.2f}) — abort episode')
              break
            if (step_idx - _pp_start) >= _pp_max_steps:
              logger.info('[pure-pursuit] step cap reached')
              break
            # arrived = within _succ_thr of path end, OR stuck near goal
            # (blocked by wall/counter — robot cannot get physically closer).
            _pos_hist.append(_cur.copy())
            if len(_pos_hist) > _stuck_win:
              _pos_hist.pop(0)
            _d_final = float(np.linalg.norm(_cur - _final_xy))
            _arrived = _d_final <= _succ_thr
            if (not _arrived) and len(_pos_hist) >= _stuck_win:
              _moved = max(float(np.linalg.norm(_cur - _p)) for _p in _pos_hist)
              if _moved < _stuck_eps:
                if _d_final <= _stuck_goal_r:
                  logger.info(f'[pure-pursuit] stuck near goal @ step {step_idx} '
                              f'(moved {_moved:.3f}m/{_stuck_win}st, d_final={_d_final:.2f}) '
                              f'— treat as arrived')
                  _arrived = True
                else:
                  # #29 REPLAN v2: stuck FAR → mark current robot position as
                  # obstacle in avoidance map (so A* won't route through this
                  # wedge spot again), then replan. Without wedge-marker, naive
                  # replan generates the same path → robot stucks again.
                  if _pp_replan_count < _pp_replan_max:
                    # ESCAPE (#34 v3 first-improvement): try each direction in
                    # turn (back, forward, left, right). Break on FIRST direction
                    # that improves d_final by >= PP_ESCAPE_IMPROVE_M (default
                    # 0.1m). If none improve, fall through with robot at final
                    # tried position. Avoids cycling-mode regression where each
                    # replan picks single (sometimes wrong) direction.
                    if os.environ.get('PP_ESCAPE_ENABLED', '1') == '1':
                      _esc_dirs = [
                          ('back',    -1.0,  0.0),
                          ('forward',  1.0,  0.0),
                          ('left',     0.0,  1.0),
                          ('right',    0.0, -1.0),
                      ]
                      _esc_speed = float(os.environ.get('PP_ESCAPE_SPEED', '0.5'))
                      _esc_steps_per_dir = int(os.environ.get('PP_ESCAPE_STEPS', '20'))
                      _esc_improve_thresh = float(os.environ.get('PP_ESCAPE_IMPROVE_M', '0.1'))
                      _d_orig = float(np.linalg.norm(_cur - _final_xy))
                      _d_baseline = _d_orig
                      _esc_chosen = None
                      for _dir_name, _esc_dx, _esc_dy in _esc_dirs:
                        _esc_vx = _esc_dx * _esc_speed
                        _esc_vy = _esc_dy * _esc_speed
                        for _es in range(_esc_steps_per_dir):
                          _esc_act = np.array([_esc_vx, _esc_vy, 0.0])
                          _ci = self._nav_controller.execute(_esc_act)
                          _ci['controller_step'] = step_idx
                          for _vc in VoxPoserRobocasa.VIDEO_RECORD_CAMERAS:
                            _k = f"{_vc}_image"
                            if _k in _ci['mp_info'][0]:
                              _ci[_k] = _ci['mp_info'][0][_k][::-1]
                          controller_infos[step_idx] = _ci
                          step_idx += 1
                        _obs = self._env.env._get_observations()
                        _cur = np.asarray(_obs['robot0_base_pos'][:2], dtype=float)
                        _d_after = float(np.linalg.norm(_cur - _final_xy))
                        _improv = _d_baseline - _d_after
                        logger.info(f'[pure-pursuit] ESCAPE dir={_dir_name}: d_final '
                                    f'{_d_baseline:.2f} → {_d_after:.2f} ({_improv:+.2f}m)')
                        if _improv >= _esc_improve_thresh:
                          _esc_chosen = _dir_name
                          logger.info(f'[pure-pursuit] ESCAPE chose {_esc_chosen} '
                                      f'(orig d={_d_orig:.2f} → {_d_after:.2f}, replan next)')
                          break
                        _d_baseline = _d_after
                      if _esc_chosen is None:
                        logger.warning(f'[pure-pursuit] ESCAPE all 4 dirs failed to improve '
                                       f'(orig {_d_orig:.2f} → {_d_baseline:.2f}); replan anyway')
                    try:
                      _robot_obs = self.detect('robot_mobile_base')
                      _cur_grid = np.asarray(_robot_obs['position'])[:2]
                      # Build augmented avoid map: copy + disk obstacle at robot pos
                      _avoid_aug = np.asarray(_cached_avoid).copy()
                      _Ha, _Wa = _avoid_aug.shape
                      _wmin = np.asarray(self._env.workspace_bounds_min[:2], float)
                      _wmax = np.asarray(self._env.workspace_bounds_max[:2], float)
                      _sx_m = (_wmax[0]-_wmin[0])/_Wa; _sy_m = (_wmax[1]-_wmin[1])/_Ha
                      _cr = int((_cur[1]-_wmin[1])/_sy_m)
                      _cc = int((_cur[0]-_wmin[0])/_sx_m)
                      _wedge_r_m = float(os.environ.get('PP_REPLAN_WEDGE_R', '0.5'))
                      _wedge_core_m = float(os.environ.get('PP_REPLAN_WEDGE_CORE', '0.15'))
                      _wedge_cells = max(1, int(_wedge_r_m / ((_sx_m+_sy_m)/2)))
                      _core_cells = max(1, int(_wedge_core_m / ((_sx_m+_sy_m)/2)))
                      _yy, _xx = np.ogrid[:_Ha, :_Wa]
                      _dist_sq = (_yy-_cr)**2 + (_xx-_cc)**2
                      # DONUT: inner core stays free (A* start unblocked),
                      # outer ring blocked (force replan to avoid wedge area).
                      _wm = (_dist_sq <= _wedge_cells**2) & (_dist_sq > _core_cells**2)
                      _avoid_aug[_wm] = 1.0
                      logger.info(f'[pure-pursuit] REPLAN v3: donut wedge marker '
                                  f'(core={_wedge_core_m}m, outer={_wedge_r_m}m, '
                                  f'cells {_core_cells}-{_wedge_cells}) at robot pos px({_cr},{_cc})')
                      _new_path, _ = self._planner.navigation_optimize(
                          _cur_grid, _cached_aff, _avoid_aug,
                          robot_radius_cells=_cached_rr)
                      if _new_path is not None and len(_new_path) > 1:
                        _new_traj = self._path2traj_navigation(
                            _new_path, _cached_avoid, _cached_rot_map, _cached_vel_map)
                        if len(_new_traj) > 1:
                          traj_world = _new_traj
                          _path_xy = np.array([np.asarray(w[0])[:2] for w in traj_world], dtype=float)
                          _final_xy = _path_xy[-1]
                          _fy = np.asarray(traj_world[-1][1])
                          _final_yaw = float(_fy.item()) if _fy.size == 1 and np.isfinite(_fy.item()) else None
                          _last_k = 0
                          _pos_hist = []
                          _pp_replan_count += 1
                          logger.info(f'[pure-pursuit] REPLAN #{_pp_replan_count} @ step '
                                      f'{step_idx} (d_final was {_d_final:.2f}m, '
                                      f'new path {len(_path_xy)} wps, '
                                      f'new end ({_final_xy[0]:.2f},{_final_xy[1]:.2f}))')
                          continue
                    except Exception as _re:
                      logger.warning(f'[pure-pursuit] replan exception: {_re}')
                  logger.warning(f'[pure-pursuit] stuck FAR from goal @ step '
                                 f'{step_idx} (d_final={_d_final:.2f}m) — abort '
                                 f'(replan {_pp_replan_count}/{_pp_replan_max} exhausted)')
                  break
            # arrived → align final yaw, then stop
            if _arrived:
              if _final_yaw is not None:
                _dy = (_final_yaw - _cyaw + np.pi) % (2 * np.pi) - np.pi
                if abs(_dy) > self._yaw_threshold:
                  _act = np.array([0.0, 0.0,
                                   float(np.clip(_dy * _kp_rot, -_omega_max, _omega_max))])
                  _ci = self._nav_controller.execute(_act)
                  _ci['controller_step'] = step_idx
                  for _vc in VoxPoserRobocasa.VIDEO_RECORD_CAMERAS:
                    _k = f"{_vc}_image"
                    if _k in _ci['mp_info'][0]:
                      _ci[_k] = _ci['mp_info'][0][_k][::-1]
                  controller_infos[step_idx] = _ci
                  step_idx += 1
                  pbar.update(1)
                  continue
              logger.info(f'[pure-pursuit] goal reached @ step {step_idx}')
              break
            # 1) project robot onto path polyline → closest segment (k, t),
            #    searched forward-only within [_last_k, _last_k+PP_PROJ_WINDOW]
            _bk, _bt, _bd = _last_k, 0.0, 1e9
            _k_hi = min(len(_path_xy) - 1, _last_k + _proj_win)
            for _k in range(_last_k, _k_hi):
              _a, _b = _path_xy[_k], _path_xy[_k + 1]
              _ab = _b - _a
              _L2 = float(_ab.dot(_ab))
              _t = 0.0 if _L2 < 1e-9 else float(np.clip((_cur - _a).dot(_ab) / _L2, 0.0, 1.0))
              _d = float(np.linalg.norm(_cur - (_a + _t * _ab)))
              if _d < _bd:
                _bk, _bt, _bd = _k, _t, _d
            _last_k = _bk   # monotonic — projection only advances
            # 2) lookahead point: walk PP_LOOKAHEAD_M of arc length forward
            _remain = _Ld
            _li, _lt = _bk, _bt
            _look = _final_xy.copy()
            while _li < len(_path_xy) - 1:
              _a, _b = _path_xy[_li], _path_xy[_li + 1]
              _seg = _b - _a
              _seglen = float(np.linalg.norm(_seg))
              _avail = _seglen * (1.0 - _lt)
              if _avail >= _remain:
                _look = _a + _seg * (_lt + _remain / max(_seglen, 1e-9))
                break
              _remain -= _avail
              _li += 1
              _lt = 0.0
            # 3) holonomic action toward lookahead.
            #    The robot is HOLONOMIC (independent vx,vy — motion-audit
            #    confirmed). It does NOT need to face its motion direction.
            #    → translation (vx,vy) follows the path in ANY direction;
            #      rotation (omega) tracks only the FINAL goal heading.
            #    Rotating toward the lookahead (classic car-like pursuit)
            #    caused runaway spin on non-90° spawns.
            #    Translation transform: STANDARD R(yaw) (audit ground truth,
            #    body+x world = (cos yaw, sin yaw), confirmed yaw 0/90/180).
            _dx, _dy = float(_look[0] - _cur[0]), float(_look[1] - _cur[1])
            _goal_yaw = _final_yaw if _final_yaw is not None else _cyaw
            _dyaw = (_goal_yaw - _cyaw + np.pi) % (2 * np.pi) - np.pi
            _omega = float(np.clip(_dyaw * _kp_rot, -_omega_max, _omega_max))
            _yawp = _cyaw + 0.5 * _omega
            _vx =  _dx * np.cos(_yawp) + _dy * np.sin(_yawp)
            _vy = -_dx * np.sin(_yawp) + _dy * np.cos(_yawp)
            _act = np.zeros(3)
            _act[0] = float(np.clip(_vx * _pp_kp, -1.0, 1.0))
            _act[1] = float(np.clip(_vy * _pp_kp, -1.0, 1.0))
            _act[2] = _omega
            # speed modulation: with a fixed 0.4m lookahead the P-term always
            # saturates → robot runs at max speed and overshoots curves into
            # walls on long big-layout paths. Slow translation when the
            # heading error is large (classic "slow down to turn").
            _turn_slow = float(os.environ.get('PP_TURN_SLOW', '100'))
            _tf = max(0.25, 1.0 - abs(_dyaw) / _turn_slow)
            _act[0] *= _tf
            _act[1] *= _tf
            # endpoint deceleration: within PP_DECEL_R of the path end, scale
            # translation down ∝ distance so the robot CONVERGES to _final_xy
            # instead of running at full speed → overshooting → orbiting it.
            _decel_r = float(os.environ.get('PP_DECEL_R', '0.4'))
            _d_end = float(np.linalg.norm(_cur - _final_xy))
            if _d_end < _decel_r:
              _de = max(0.20, _d_end / _decel_r)
              _act[0] *= _de
              _act[1] *= _de
            # per-step debug log (every 25 steps + first 6)
            _pps = step_idx - _pp_start
            if _pps < 6 or _pps % 25 == 0:
              logger.info(f'[pp s{_pps}] cur=({_cur[0]:.2f},{_cur[1]:.2f}) '
                          f'yaw={np.degrees(_cyaw):.0f} k={_bk}/{len(_path_xy)-1} '
                          f'look=({_look[0]:.2f},{_look[1]:.2f}) '
                          f'd=({_dx:+.2f},{_dy:+.2f}) gyaw={np.degrees(_goal_yaw):.0f} '
                          f'act=[{_act[0]:+.2f},{_act[1]:+.2f},{_act[2]:+.2f}]')
            _ci = self._nav_controller.execute(_act)
            _ci['controller_step'] = step_idx
            _ci['target_waypoint'] = (_look, _final_yaw if _final_yaw is not None else _cyaw, 1.0)
            for _vc in VoxPoserRobocasa.VIDEO_RECORD_CAMERAS:
              _k = f"{_vc}_image"
              if _k in _ci['mp_info'][0]:
                _ci[_k] = _ci['mp_info'][0][_k][::-1]
            controller_infos[step_idx] = _ci
            step_idx += 1
            pbar.update(1)
          pbar.close()
          step_info['controller_infos'] = controller_infos
          execute_info.append(step_info)
          continue   # skip the legacy wp-iteration loop below
        # === End PURE-PURSUIT mode ===

        while i < len(traj_world):
          waypoint = traj_world[i]
          waypoint_reach = False
          is_last = (i == len(traj_world) - 1)
          dist_threshold = self._dist_threshold
          wp_step = 0
          _replan_jump = False
          # translate_only mode: explicit 2-phase per waypoint.
          #   Phase A: rotate in place toward waypoint until aligned
          #   Phase B: translate forward only until at waypoint
          # No mode switching once Phase B starts — committed to translation.
          # Avoids the rotate-drift-rotate cycle that stuck Person G v3.
          if self._nav_mode == 'translate_only':
            goal_xy_wp = np.asarray(waypoint[0])
            # B' (k-lookahead): rotate toward wp[i+k] for robust heading.
            # wp[i] may be nearly cur_pos (noisy direction); wp[i+k] gives
            # a stable target direction. Phase B still terminates at wp[i].
            _phase_a_k = int(os.environ.get('PHASE_A_LOOKAHEAD_K', '0'))
            if _phase_a_k > 0:
              _la_idx = min(i + _phase_a_k, len(traj_world) - 1)
              rot_target_xy = np.asarray(traj_world[_la_idx][0])
            else:
              rot_target_xy = goal_xy_wp
            # === Phase A: ROTATE ===
            ROT_TOL = np.deg2rad(8)
            ROT_KP = 3.0
            OMEGA_MAX = 0.3
            ROT_MAX_STEPS = 60
            for _rs in range(ROT_MAX_STEPS):
              _cp = self._env.env._get_observations()['robot0_base_pos'][:2]
              _q2 = self._env.env._get_observations()['robot0_base_quat']
              _cy2 = quat2euler([_q2[3], _q2[0], _q2[1], _q2[2]])[2]
              # body +x faces (dx, dy) when real_yaw = atan2(dy, -dx) per R(π-yaw)
              _t_yaw = float(np.arctan2(rot_target_xy[1] - _cp[1], -(rot_target_xy[0] - _cp[0])))
              _d = (_t_yaw - _cy2 + np.pi) % (2*np.pi) - np.pi
              if abs(_d) <= ROT_TOL:
                logger.debug(f"wp{i} phase-A done in {_rs} steps (delta={np.degrees(_d):+.1f}deg)")
                break
              act = np.array([0.0, 0.0, float(np.clip(_d * ROT_KP, -OMEGA_MAX, OMEGA_MAX))])
              ci = self._nav_controller.execute(act)
              ci['controller_step'] = step_idx
              ci['target_waypoint'] = waypoint
              for _vc in VoxPoserRobocasa.VIDEO_RECORD_CAMERAS:
                _k = f"{_vc}_image"
                if _k in ci['mp_info'][0]:
                  ci[_k] = ci['mp_info'][0][_k][::-1]
              controller_infos[step_idx] = ci
              step_idx += 1
            # === Phase B: TRANSLATE ===
            TRANS_KP = 5.0
            VX_MAX = 1.0
            TRANS_MAX_STEPS = self._max_steps_per_waypoint * (3 if is_last else 1)
            wp_reach_thr = 0.5 if is_last else 0.10
            for _ts in range(TRANS_MAX_STEPS):
              _cp = self._env.env._get_observations()['robot0_base_pos'][:2]
              _dist_now = float(np.hypot(goal_xy_wp[0] - _cp[0], goal_xy_wp[1] - _cp[1]))
              if _dist_now <= wp_reach_thr:
                logger.debug(f"wp{i} phase-B done in {_ts} steps (dist={_dist_now:.3f}m)")
                waypoint_reach = True
                break
              # Phase B action: default = body +x only (v1 baseline).
              # PHASE_B_HOLONOMIC=1 → use both body +x and +y (R(-yaw) on world Δ).
              # Allows Phase B to recover lateral drift from Phase A rotation.
              if os.environ.get('PHASE_B_HOLONOMIC', '0') == '1':
                _q2 = self._env.env._get_observations()['robot0_base_quat']
                _yaw_now = quat2euler([_q2[3], _q2[0], _q2[1], _q2[2]])[2]
                _dx_w = goal_xy_wp[0] - _cp[0]
                _dy_w = goal_xy_wp[1] - _cp[1]
                # Controller's body→world is R(π - yaw). World→body inverse:
                _vx_body = -_dx_w * np.cos(_yaw_now) + _dy_w * np.sin(_yaw_now)
                _vy_body = -_dx_w * np.sin(_yaw_now) - _dy_w * np.cos(_yaw_now)
                act = np.array([
                  float(np.clip(_vx_body * TRANS_KP, -VX_MAX, VX_MAX)),
                  float(np.clip(_vy_body * TRANS_KP, -VX_MAX, VX_MAX)),
                  0.0,
                ])
              else:
                act = np.array([float(np.clip(_dist_now * TRANS_KP, 0.0, VX_MAX)), 0.0, 0.0])
              ci = self._nav_controller.execute(act)
              ci['controller_step'] = step_idx
              ci['target_waypoint'] = waypoint
              for _vc in VoxPoserRobocasa.VIDEO_RECORD_CAMERAS:
                _k = f"{_vc}_image"
                if _k in ci['mp_info'][0]:
                  ci[_k] = ci['mp_info'][0][_k][::-1]
              controller_infos[step_idx] = ci
              step_idx += 1
            # advance to next waypoint regardless of reach status
            i += 1
            pbar.update(1)
            continue
          # holonomic mode (original): mixed v_x/v_y/omega
          lookahead_idx = min(i + 2, len(traj_world) - 1)
          lookahead_wp = traj_world[lookahead_idx]
          while not waypoint_reach:
            if is_last:
              traj_action, dist_to_yaw = self._navigate_to_trajectory(traj_world[i], traj_world[i], lookahead_wp=lookahead_wp)
            else:
              traj_action, dist_to_yaw = self._navigate_to_trajectory(traj_world[i], traj_world[i+1], lookahead_wp=lookahead_wp)
            controller_info = self._nav_controller.execute(traj_action)
            cur_pos = self._env.env._get_observations()['robot0_base_pos']
            _q = self._env.env._get_observations()['robot0_base_quat']  # robosuite xyzw
            cur_yaw = quat2euler([_q[3], _q[0], _q[1], _q[2]])[2]       # → wxyz, [2]=yaw_Z
            dxy = cur_pos[:2] - waypoint[0]

            controller_info['controller_step'] = step_idx
            controller_info['target_waypoint'] = waypoint
            # Capture every VLM-input camera viewpoint for downstream review.
            for _vlm_cam in VoxPoserRobocasa.VIDEO_RECORD_CAMERAS:
                _key = f"{_vlm_cam}_image"
                if _key in controller_info['mp_info'][0]:
                    controller_info[_key] = controller_info['mp_info'][0][_key][::-1]
            controller_infos[step_idx] = controller_info
            step_idx += 1

            # Global cap: emergency exit if total steps exceed budget
            # (replan-loop oscillation safeguard).
            if (step_idx - _global_start_step) >= _global_max_steps:
              logger.warning(f'global step cap reached ({_global_max_steps}); exiting outer loop')
              waypoint_reach = True
              break

            # === REPLAN: rerun A* from cur_pos every N steps ===
            if _replan_n > 0 and step_idx % _replan_n == 0:
              try:
                _robot_obs = self.detect('robot_mobile_base')
                _cur_grid = np.asarray(_robot_obs['position'])[:2]
                _new_path, _ = self._planner.navigation_optimize(
                    _cur_grid, _cached_aff, _cached_avoid,
                    robot_radius_cells=_cached_rr)
                if _new_path is not None and len(_new_path) > 1:
                  _new_traj = self._path2traj_navigation(
                      _new_path, _cached_avoid, _cached_rot_map, _cached_vel_map)
                  # Skip first wp (= cur_pos) so loop targets the next wp
                  # ahead instead of standing still trying to "reach" cur_pos.
                  if len(_new_traj) > 1:
                    traj_world = _new_traj[1:self._cfg['num_waypoints_per_plan']+1]
                    logger.info(f'replan @ step {step_idx}: {len(traj_world)} wps (skipped wp[0]=cur_pos)')
                    i = 0
                    _replan_jump = True
                    break  # exit inner while, restart outer with new traj
              except Exception as _rpe:
                logger.warning(f'replan failed: {_rpe}')
            # ===================================================

            # ---- Y1 debug log: per-step state on LAST wp only (single L6 sweep) ----
            if is_last:
              try:
                _last_yaw_log = np.asarray(waypoint[1]).item() if np.asarray(waypoint[1]).size == 1 else float('nan')
                _delta = (_last_yaw_log - cur_yaw + np.pi) % (2 * np.pi) - np.pi if not np.isnan(_last_yaw_log) else 0.0
                with open('/tmp/yaw_debug_last_wp.log', 'a') as _lf:
                  _lf.write(f"step={step_idx} dist={float(np.linalg.norm(dxy)):.4f} cur_yaw={cur_yaw:.4f} goal_yaw={_last_yaw_log:.4f} delta={_delta:.4f} act={traj_action.tolist()}\n")
              except Exception:
                pass
            # -----------------------------------------------------------------------

            # ---- Y4 fix: last waypoint exit also accepts position-only success ----
            # Last wp is intentionally placed inside the affordance disk which
            # may overlap a counter/fixture (planner picks the cell closest to
            # goal). Robot may never reach within dist_threshold=5cm because the
            # mesh blocks it ~30cm out, so it pushes into the wall for 300 steps
            # → physics torque rotates the base randomly. Allow early exit when
            # (a) dist <= success threshold (0.5m) AND yaw error within tolerance,
            # or (b) tight dist (5cm) like before for non-last waypoints.
            _dist_now = float(np.linalg.norm(dxy))
            if is_last:
              # Use config yaw_threshold (0.35 rad ≈ 20°)
              _last_yaw_chk = np.asarray(waypoint[1]).item() if np.asarray(waypoint[1]).size == 1 else float('nan')
              if np.isnan(_last_yaw_chk):
                _yaw_ok = True   # no rotation requirement
              else:
                _yaw_err = abs((_last_yaw_chk - cur_yaw + np.pi) % (2 * np.pi) - np.pi)
                _yaw_ok = _yaw_err < self._yaw_threshold
              if _dist_now <= 0.5 and _yaw_ok:
                logger.debug(f"last waypoint reached (dist={_dist_now:.3f}m, yaw_ok={_yaw_ok})")
                waypoint_reach = True
                break
            else:
              if _dist_now <= dist_threshold:
                waypoint_reach = True
                break
            # ---------------------------------------------------------------------

            wp_step += 1
            # Give the LAST waypoint extra time so yaw can fully align
            # (success criterion needs <36.9° but config yaw_threshold is
            # 20° — at clipped rotation rate ~0.5°/step, 90° rotation needs
            # ~180 steps, exceeding the default 100).
            wp_max = self._max_steps_per_waypoint * (3 if is_last else 1)
            if wp_step >= wp_max:
              logger.debug(f"waypoint {i} exceeded {wp_max} steps (dist={np.linalg.norm(dxy):.3f}), skipping")
              break

          # Outer while loop control: if replan reset i, don't increment
          if not _replan_jump:
            i += 1
            pbar.update(1)
        pbar.close()
        step_info['controller_infos'] = controller_infos
        execute_info.append(step_info)
        curr_pos = movable_obs['position'][:2].astype(int)
        if distance_transform_edt(1 - _affordance_map)[tuple(curr_pos)] <= 2:
          logger.info(f'[{get_clock_time()}] reached target; terminating')
          break
    # Dump planner state for offline visualisation. We strip the heavy
    # `controller_infos` (mp_info has full sim state + camera images) and
    # only persist arrays that scripts/visualize_voxposer_task.py renders.
    try:
      from transforms3d.euler import quat2euler as _q2e
      _dump_iters = []
      for _si in execute_info:
        _wps = _si.get("traj_world") or []
        _wp_xy = np.asarray([np.asarray(w[0])[:3] for w in _wps]) if _wps else np.empty((0, 3))
        # Recover key-waypoint yaw from the rotation entry _w[1].
        # Navigation: scalar yaw (radians). NaN means "no explicit rotation
        #   requirement"; matching controller policy, we substitute the
        #   direction toward the NEXT waypoint (seg_dir) so viz arrows show
        #   the robot facing where it's about to travel. Last wp inherits
        #   prior wp's heading (no next wp to face).
        # Manipulation: wxyz quaternion (4-element).
        _wp_yaw = []
        n_wps = len(_wps)
        for i, _w in enumerate(_wps):
          try:
            _r = np.asarray(_w[1])
            if _r.ndim == 0 or _r.size == 1:
              v = float(_r)
              if np.isnan(v):
                # NaN sentinel → seg_dir to next wp (or prior heading if last)
                if i < n_wps - 1:
                  _w_now = np.asarray(_wp_xy[i])[:2]
                  _w_next = np.asarray(_wp_xy[i + 1])[:2]
                  _d = _w_next - _w_now
                  if np.linalg.norm(_d) > 1e-6:
                    v = float(np.arctan2(_d[1], _d[0]))
                  else:
                    v = _wp_yaw[-1] if _wp_yaw else 0.0
                else:
                  v = _wp_yaw[-1] if _wp_yaw else 0.0
            else:
              v = float(_q2e(_r)[2])  # [2] = yaw_Z; not [0]=roll (legacy bug)
            _wp_yaw.append(v)
          except Exception:
            _wp_yaw.append(0.0)
        _dump_iters.append({
          "plan_iter":     int(_si.get("plan_iter", 0)),
          "start_pos":     np.asarray(_si.get("start_pos")),
          "path_pixel":    np.asarray(_si.get("path_pixel")),
          "traj_world":    _wp_xy,
          "traj_world_yaw": np.asarray(_wp_yaw, dtype=np.float32),
          "affordance_map": np.asarray(_si.get("affordance_map")),
          "avoidance_map":  np.asarray(_si.get("avoidance_map")),
          "rotation_map":   np.asarray(_si.get("rotation_map")),
          "velocity_map":   np.asarray(_si.get("velocity_map")),
        })
      _dump_path = os.path.join(self._output_dir, "voxposer_dump.npz")
      # Project workspace_bounds 4 corners to topview UV pixels so the
      # visualiser can warp planner maps onto the kitchen floor exactly
      # (no floor-mask heuristic needed).
      _ws_min = np.asarray(self._env.workspace_bounds_min)
      _ws_max = np.asarray(self._env.workspace_bounds_max)
      _topview_corners_uv = None
      try:
        _sim = self._env.env.sim
        _cid = _sim.model.camera_name2id('topview')
        _cam_pos = _sim.data.cam_xpos[_cid].copy()
        _cam_mat = _sim.data.cam_xmat[_cid].reshape(3, 3).copy()
        _fovy = float(_sim.model.cam_fovy[_cid])
        _W = int(self._env.cam_width)
        _H = int(self._env.cam_height)
        _fy = (_H / 2.0) / np.tan(np.radians(_fovy / 2.0))
        _fx = _fy
        # Project corners onto the ACTUAL floor plane. workspace_bounds_min[2]
        # is the body-bbox lower z (e.g. -1.0m underground), NOT the floor z.
        # Read the real floor z from a floor geom's xpos. Falls back to 0.0
        # only if no floor geoms are tracked (legacy envs).
        _floor_z = 0.0
        try:
            _fids = getattr(self._env, 'floor_mask_ids', [])
            if _fids:
                _floor_z = float(_sim.data.geom_xpos[_fids[0], 2])
        except Exception:
            pass
        _corners_world = np.array([
            [_ws_min[0], _ws_min[1], _floor_z],  # planner (0, 0)
            [_ws_min[0], _ws_max[1], _floor_z],  # planner (0, map_size-1)
            [_ws_max[0], _ws_max[1], _floor_z],  # planner (map_size-1, map_size-1)
            [_ws_max[0], _ws_min[1], _floor_z],  # planner (map_size-1, 0)
        ])
        _corners_uv = []
        for _w in _corners_world:
            _rel = (_w - _cam_pos)
            _cam_frame = _cam_mat.T @ _rel
            _depth = -_cam_frame[2]
            _u = (_W / 2.0) + _fx * (_cam_frame[0] / _depth)
            _v = (_H / 2.0) - _fy * (_cam_frame[1] / _depth)
            _corners_uv.append([float(_u), float(_v)])
        _topview_corners_uv = np.asarray(_corners_uv)
      except Exception as _proj_err:
        logger.warning(f"topview corner projection failed: {_proj_err}")

      # Best-effort: locate the obstacle in world xy. Different task types
      # surface this in different attributes / fixtures, so try a few.
      #
      # Order (corrected 2026-05-07):
      #   1. obstacle_name == "human" → _get_human_pos() (posed_human fixture is the body)
      #   2. Floor-placed obstacles (cat/dog/kettlebell/vase/crawling_baby):
      #      look up the actual MuJoCo body whose name starts with "obstacle"
      #      and use the geom_xpos centroid. Robocasa places these via
      #      sample_region_kwargs with size=(0.8, 0.8), so they sit anywhere
      #      up to ~40cm from `_obstacle_blocking_xy` → that planning anchor
      #      should NOT be reported as the obstacle location.
      #   3. Fall back to `_obstacle_blocking_xy` (only when (2) finds no
      #      body — e.g. table-mounted drinks where the body is on a fixture
      #      and the planning anchor matches well enough).
      _obstacle_xy = None
      _obstacle_name = None
      try:
        _kitchen = getattr(self._env, "env", None)
        _obstacle_name = getattr(_kitchen, "obstacle", None) if _kitchen else None
        # Capture cat position at DUMP-SAVE time (= during LMP planning,
        # matches what LLM detected via parse_query_obj('cat'). The earlier
        # _initial_obstacle_xy snapshot at load_task time drifted vs LLM's
        # cat (~32cm in L8) due to physics steps between load_task and
        # avoidance map generation.
        if _obstacle_name == "human" and hasattr(self._env, "_get_human_pos"):
          _p = self._env._get_human_pos()
          if _p is not None:
            _obstacle_xy = np.asarray(_p)[:2]
        if _obstacle_xy is None and _kitchen is not None:
          _sim2 = _kitchen.sim
          for _bid in range(_sim2.model.nbody):
            _nm = _sim2.model.body_id2name(_bid) or ""
            if _nm.startswith("obstacle"):
              # body_xpos is the body's reference frame; the actual mesh can
              # be attached with an offset. Use the mean of geom_xpos for all
              # geoms attached so the marker lands on the rendered mesh, not
              # on the body anchor.
              _gxs = []
              for _gid in range(_sim2.model.ngeom):
                if _sim2.model.geom_bodyid[_gid] == _bid:
                  _gxs.append(_sim2.data.geom_xpos[_gid])
              if _gxs:
                _obstacle_xy = np.mean(_gxs, axis=0)[:2]
              else:
                _obstacle_xy = np.asarray(_sim2.data.body_xpos[_bid])[:2]
              break
        # Final fallback: planning anchor (only used when no obstacle body
        # exists in the scene — e.g. corrupt task or unusual config).
        if _obstacle_xy is None and _kitchen is not None:
          for _attr in ("_obstacle_blocking_xy", "_obstacle_xy",
                        "obstacle_pos", "_obstacle_pos"):
            _v = getattr(_kitchen, _attr, None)
            if _v is not None:
              _obstacle_xy = np.asarray(_v).reshape(-1)[:2]
              break

        # Same correction for goal_xy: store the actual visible mesh
        # centroid of the destination fixture so the goal marker matches
        # what's rendered. We pull it from the 3-camera point cloud the LMP
        # already computes (parse_query_obj path), with body_xpos as
        # fallback. This is a no-op for cases where the two coincide.
        _goal_xy = None
        try:
          _tgt = getattr(_kitchen, "target_fixture", None)
          if _tgt is not None and hasattr(_tgt, "pos"):
            _goal_xy = np.asarray(_tgt.pos)[:2]
        except Exception:
          pass
      except Exception as _ob_err:
        logger.warning(f"obstacle position lookup failed: {_ob_err}")

      _save_kwargs = dict(
          iters=np.array(_dump_iters, dtype=object),
          workspace_bounds_min=_ws_min,
          workspace_bounds_max=_ws_max,
          map_size=np.asarray(self._map_size),
          map_h=np.asarray(self._map_h),  # rectangular grid (rows = y cells)
          map_w=np.asarray(self._map_w),  # rectangular grid (cols = x cells)
      )
      if _topview_corners_uv is not None:
          _save_kwargs["topview_corners_uv"] = _topview_corners_uv
      if _obstacle_xy is not None:
          _save_kwargs["obstacle_xy"] = np.asarray(_obstacle_xy, dtype=np.float32)
      if _obstacle_name:
          _save_kwargs["obstacle_name"] = np.asarray(str(_obstacle_name))
      # Also persist the obstacle position captured at load_task time
      # (= same instant initial_topview.png is rendered). The viz uses this
      # for the cat MARKER so it aligns with the cat as seen in the topview;
      # `obstacle_xy` (LMP-planning-time) is preserved for halo / analysis.
      try:
        _init_obs = getattr(self._env, "_initial_obstacle_xy", None)
        if _init_obs is not None:
          _save_kwargs["obstacle_xy_init"] = np.asarray(_init_obs, dtype=np.float32)
      except Exception:
        pass
      if _goal_xy is not None:
          _save_kwargs["goal_xy_fixture"] = np.asarray(_goal_xy, dtype=np.float32)
      np.savez_compressed(_dump_path, **_save_kwargs)
      logger.info(f"voxposer dump → {_dump_path}")
    except Exception as _dump_err:
      logger.warning(f"voxposer dump failed: {_dump_err}")
    # Save one mp4 per recorded camera so downstream review/labeling can
    # reproduce exactly what the VLM sees at inference. Source of truth:
    # VoxPoserRobocasa.VIDEO_RECORD_CAMERAS.
    #
    # IMPORTANT: disable run_LMP.py's TASK_TIMEOUT_SEC alarm BEFORE video
    # encoding. Sim+LMP work is already done by this point; the only thing
    # left is mp4 writing + ffmpeg re-encode which can take 60-300s for 5
    # cameras × 3000+ frames. Without disabling, long sims (e.g. L0
    # ~3000 steps) hit timeout DURING video save → 3 retries × timeout
    # all fail (TaskTimeout raised in cv2 loop) → no results.json saved
    # for that task. After this block we just `return execute_info`, so no
    # subsequent operation needs timeout protection.
    try:
      import signal as _signal
      _signal.alarm(0)   # disable task timeout — video save can take its time
    except Exception:
      pass  # not on Unix or alarm not supported — proceed without disable
    if controller_infos:
        for _cam in VoxPoserRobocasa.VIDEO_RECORD_CAMERAS:
            _kw = f"{_cam}_image"
            # Only save if at least one frame actually has this camera.
            if any(_kw in v for v in controller_infos.values()):
                save_video_images(
                    controller_infos,
                    keyword=_kw,
                    save_path=os.path.join(self._output_dir, f"{_kw}.mp4"),
                )
    logger.info(f'[{get_clock_time()}] finished executing navigation')
    return execute_info
  
  def yaw_toward(self, from_pos, to_pos):
    """Compute yaw angle (radians) from from_pos toward to_pos.

    Args:
      from_pos: [x, y, ...] source position
      to_pos: [x, y, ...] target position
    Returns:
      float: yaw in radians
    """
    dx = float(to_pos[0]) - float(from_pos[0])
    dy = float(to_pos[1]) - float(from_pos[1])
    return float(np.arctan2(dy, dx))

  def cm2index(self, cm, direction):
    if isinstance(direction, str) and direction == 'x':
      x_resolution = self._resolution[0] * 100  # resolution is in m, we need cm
      return int(cm / x_resolution)
    elif isinstance(direction, str) and direction == 'y':
      y_resolution = self._resolution[1] * 100
      return int(cm / y_resolution)
    elif isinstance(direction, str) and direction == 'z':
      z_resolution = self._resolution[2] * 100
      return int(cm / z_resolution)
    else:
      # calculate index along the direction
      if not isinstance(direction, np.ndarray):
        direction = np.array(direction, dtype=float)
      if direction.shape == (2,):
        direction = np.append(direction, 0.0)
      assert direction.shape == (3,), f"cm2index: expected 3D direction, got shape {direction.shape}"
      direction = normalize_vector(direction)
      x_cm = cm * direction[0]
      y_cm = cm * direction[1]
      z_cm = cm * direction[2]
      x_index = self.cm2index(x_cm, 'x')
      y_index = self.cm2index(y_cm, 'y')
      z_index = self.cm2index(z_cm, 'z')
      # For NAVIGATION, position is (row=y, col=x, z). Return offsets in
      # the same (row, col, z) order so direct addition `position + offset`
      # by LLM produces a valid array index.
      if bool(getattr(self._env, 'navigate_task', False)):
        return np.array([y_index, x_index, z_index])
      return np.array([x_index, y_index, z_index])
  
  def index2cm(self, index, direction=None):
    if direction is None:
      average_resolution = np.mean(self._resolution)
      return index * average_resolution * 100  # resolution is in m, we need cm
    elif direction == 'x':
      x_resolution = self._resolution[0] * 100
      return index * x_resolution
    elif direction == 'y':
      y_resolution = self._resolution[1] * 100
      return index * y_resolution
    elif direction == 'z':
      z_resolution = self._resolution[2] * 100
      return index * z_resolution
    else:
      raise NotImplementedError
    
  def pointat2quat(self, vector):
    assert isinstance(vector, np.ndarray) and vector.shape == (3,), f'vector: {vector}'
    return pointat2quat(vector)

  def set_voxel_by_radius(self, voxel_map, voxel_xyz, radius_cm=0, value=1):
    """given a 3D np array, set the value of the voxel at voxel_xyz to value. If radius is specified, set the value of all voxels within the radius to value."""
    voxel_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = value
    if radius_cm > 0:
      radius_x = self.cm2index(radius_cm, 'x')
      radius_y = self.cm2index(radius_cm, 'y')
      radius_z = self.cm2index(radius_cm, 'z')
      # simplified version - use rectangle instead of circle (because it is faster)
      min_x = max(0, voxel_xyz[0] - radius_x)
      max_x = min(self._map_size, voxel_xyz[0] + radius_x + 1)
      min_y = max(0, voxel_xyz[1] - radius_y)
      max_y = min(self._map_size, voxel_xyz[1] + radius_y + 1)
      min_z = max(0, voxel_xyz[2] - radius_z)
      max_z = min(self._map_size, voxel_xyz[2] + radius_z + 1)
      voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
    return voxel_map
  
  def set_pixel_by_radius(self, pixel_map, pixel_xy_or_obj, radius_cm=0, value=1, gradient=None):
    """Set `value` over a region of `pixel_map`. Two modes:

    Object mode (preferred for fixtures and grouped objects):
        pixel_xy_or_obj is an Observation dict with `occupancy_map`. The
        actual obstacle geometry is dilated by `radius_cm` cells, so the
        avoidance halo follows the real mesh — not a centroid disk that
        collapses to floor-center for perimeter-distributed obstacles
        (counter, kitchen group, etc.).

    Robot self-skip:
        If the obj name contains 'robot' / 'mobile_base' / 'gripper',
        the call is a NO-OP. LMP composer sometimes adds avoidance halo
        around the robot itself (e.g., `at least 50cm from robot_mobile_base`)
        which makes the planner unable to find a path because the robot's
        start position is inside the halo. We skip these entirely.

    Point mode (legacy):
        pixel_xy_or_obj is [x, y] (or [x, y, z]). Sets a square of size
        2·radius_cm centred at that point.

    Gradient (optional, default False):
        If True, values inside the affected region decay linearly from
        `value` at the centre (mesh interior or point) down to 0.0 at
        the outer edge (radius_cm boundary). Useful for affordance maps
        so the planner is pulled toward the centroid rather than stopping
        at the halo edge. Other maps (avoidance, velocity, rotation)
        keep the uniform default behaviour.
    """
    if pixel_map is None or pixel_xy_or_obj is None:
        return pixel_map

    # Infer gradient from value: full-value calls (value=1, used by affordance
    # & avoidance composer code) keep gradient=True so planner is pulled toward
    # centroid / soft obstacle boundary. Partial-value calls (value<1, only
    # composer's velocity_map uses these — "30% speed in slow zone") use
    # gradient=False so the slow zone is FLAT at `value` rather than a gradient
    # 0→value (the gradient mode crawls robot at 5-10% near obstacles → stuck).
    # Caller may still pass gradient=True/False to override.
    if gradient is None:
      gradient = (float(value) >= 0.999)

    # Skip robot self-avoidance: LLM-generated avoidance for the robot
    # itself produces a halo that traps the planner (start cell already
    # inside avoidance). Detect by name tokens.
    try:
      _name_lower = (pixel_xy_or_obj.get('name', '') if isinstance(pixel_xy_or_obj, dict)
                     else getattr(pixel_xy_or_obj, 'name', '') or '').lower()
    except Exception:
      _name_lower = ''
    if any(k in _name_lower for k in ('robot', 'mobile_base', 'gripper')):
      logger.info(f"[set_pixel_by_radius] SKIP robot self-avoidance: name='{_name_lower}'")
      return pixel_map

    # Duck-typed object detection. Supports Observation (dict subclass),
    # DynamicObservation, IterableDynamicObservation, and raw [x,y].
    occ = None
    try:
      candidate = pixel_xy_or_obj.occupancy_map  # __getattr__ delegates for dynamic types
      if candidate is not None:
        occ = np.asarray(candidate)
    except (AttributeError, KeyError, TypeError):
      occ = None

    # pixel_map may be VoxelIndexingWrapper without a .shape attr;
    # underlying ndarray exposes .shape correctly.
    pm_arr = pixel_map.array if hasattr(pixel_map, 'array') else pixel_map
    pm_shape = pm_arr.shape

    # Prefer building occupancy directly from world point cloud (skips the
    # voxel/pixel index swap entirely). Falls back to the legacy voxel
    # occupancy_map only if world points unavailable.
    pc_world = None
    try:
      _pc = pixel_xy_or_obj._point_cloud_world
      if _pc is not None and len(_pc):
        pc_world = np.asarray(_pc)
    except (AttributeError, KeyError, TypeError):
      pc_world = None
    if pc_world is not None:
      occ = pc2pixel_map(
          pc_world.astype(np.float32),
          self._env.workspace_bounds_min,
          self._env.workspace_bounds_max,
          pm_shape[0], pm_shape[1],
      ).astype(bool)
    elif occ is not None and occ.size > 0:
      if occ.ndim == 3:
        occ = occ.any(axis=2)
      occ = occ.astype(bool)
      # Legacy voxel_map convention is (x_idx, y_idx). pixel_map is
      # (row=y, col=x). Transpose before resize.
      occ = occ.T
      if occ.shape != pm_shape:
        try:
          import cv2
          occ = cv2.resize(occ.astype(np.uint8),
                           (pm_shape[1], pm_shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        except Exception:
          occ = None
    if occ is not None and occ.any():
      # Merge multi-part fixtures: bridge small gaps between sub-geoms of
      # the same physical object (e.g. main_door = main + door + handle +
      # trims; window = frame + glass; sink = basin + faucet) so the
      # radius dilation operates on a single continuous mask. cv2's
      # morphologyEx CLOSE is 3-10× faster than scipy.binary_closing
      # because OpenCV's C implementation handles large kernels natively
      # (single pass per op instead of N iterations of 3x3 dilation+erode).
      import cv2 as _cv2
      _cell_m = float(self._resolution[0])
      _gap_cells = max(1, int(round(0.25 / _cell_m)))
      occ_pre = occ
      try:
        # Kernel size = (2*_gap_cells+1) for symmetric closing.
        _ksz = 2 * _gap_cells + 1
        _kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (_ksz, _ksz))
        _occ_u8 = occ.astype(np.uint8)
        _closed = _cv2.morphologyEx(_occ_u8, _cv2.MORPH_CLOSE, _kernel)
        occ = _closed.astype(bool)
      except Exception:
        # Fallback to scipy if cv2 ever fails (e.g., empty mask edge case)
        from scipy.ndimage import binary_closing
        occ = binary_closing(occ_pre, iterations=_gap_cells)
      # Robot avoidance: never apply gradient. Robot body is a hard
      # boundary, not a soft falloff. Detect by name token.
      try:
        _obj_name_lower = (pixel_xy_or_obj.get('name', '') if isinstance(pixel_xy_or_obj, dict)
                           else getattr(pixel_xy_or_obj, 'name', '') or '').lower()
      except Exception:
        _obj_name_lower = ''
      if any(k in _obj_name_lower for k in ('robot', 'mobile_base', 'gripper')):
        gradient = False
      if radius_cm > 0:
        radius_cells = max(1, int(round(radius_cm / (_cell_m * 100))))
        # cv2.distanceTransform is 2-5× faster than scipy.distance_transform_edt.
        # Input: 0 = obstacle pixel, 255 = free pixel. Output: distance to
        # nearest 0 in pixel units (matches edt(~occ) semantics).
        try:
          _free_u8 = ((~occ).astype(np.uint8)) * 255
          dt_occ = _cv2.distanceTransform(_free_u8, _cv2.DIST_L2, 5)
        except Exception:
          from scipy.ndimage import distance_transform_edt
          dt_occ = distance_transform_edt(~occ)
        halo = dt_occ <= radius_cells
      else:
        halo = occ
        dt_occ = None
      target = pixel_map.array if hasattr(pixel_map, 'array') else pixel_map
      if gradient and radius_cm > 0:
        # Linear decay: value at mesh (dt=0) → 0 at halo edge (dt=radius_cells)
        decay = np.clip(1.0 - dt_occ.astype(np.float32) / float(radius_cells), 0.0, 1.0)
        target[halo] = value * decay[halo]
      else:
        target[halo] = value
      try:
        _name = pixel_xy_or_obj.get('name', '?') if isinstance(pixel_xy_or_obj, dict) else getattr(pixel_xy_or_obj, 'name', '?')
      except Exception:
        _name = '?'
      logger.info(f"[set_pixel_by_radius OCC-MODE] obj={_name} occ_cells_raw={int(occ_pre.sum())} occ_cells_merged={int(occ.sum())} halo_cells={int(halo.sum())} radius_cm={radius_cm}")
      return pixel_map

    # No usable occupancy_map → fall back to position. Prefer
    # `_position_world` (world coords) over `position` (legacy voxel
    # coords with x/y swap) — converting world → (row, col) directly
    # gives the geometrically correct cell with no swap fix-ups.
    pos_world = None
    try:
      candidate = pixel_xy_or_obj._position_world
      if candidate is not None:
        pos_world = np.asarray(candidate)[:2]
    except (AttributeError, KeyError, TypeError):
      pos_world = None

    if pos_world is not None:
      ws_min = self._env.workspace_bounds_min[:2]
      ws_max = self._env.workspace_bounds_max[:2]
      rng = ws_max - ws_min
      col = int((pos_world[0] - ws_min[0]) / rng[0] * (pm_shape[1] - 1))
      row = int((pos_world[1] - ws_min[1]) / rng[1] * (pm_shape[0] - 1))
      row = max(0, min(pm_shape[0] - 1, row))
      col = max(0, min(pm_shape[1] - 1, col))
      logger.info(f"[set_pixel_by_radius PT-MODE world] xy={pos_world.tolist()} → (row={row}, col={col}) radius_cm={radius_cm}")
      pixel_map[row, col] = value
      if radius_cm > 0:
        radius_x = self.cm2index(radius_cm, 'x')
        radius_y = self.cm2index(radius_cm, 'y')
        r0 = max(0, row - radius_y)
        r1 = min(pm_shape[0], row + radius_y + 1)
        c0 = max(0, col - radius_x)
        c1 = min(pm_shape[1], col + radius_x + 1)
        pixel_map[r0:r1, c0:c1] = value
      return pixel_map

    # Last-resort: raw [row, col] (legacy LLM code passes pixel coords
    # directly). Use as-is.
    try:
      pixel_xy = [pixel_xy_or_obj[0], pixel_xy_or_obj[1]]
    except (TypeError, KeyError, IndexError):
      return pixel_map
    # Guard against None / non-numeric coords. Caller can pass an Observation
    # whose [0]/[1] indexing returns None (parse_query_obj fallback when the
    # object isn't visible in any camera). int(None) raises TypeError —
    # silently drop those calls instead of crashing the whole episode.
    if pixel_xy[0] is None or pixel_xy[1] is None:
      logger.warning(f"[set_pixel_by_radius PT-MODE raw] skipped (None coord — likely hallucinated obstacle) xy={pixel_xy} radius_cm={radius_cm}")
      return pixel_map
    try:
      _r, _c = int(pixel_xy[0]), int(pixel_xy[1])
    except (TypeError, ValueError):
      logger.warning(f"[set_pixel_by_radius PT-MODE raw] skipped (non-numeric — likely hallucinated obstacle) xy={pixel_xy} radius_cm={radius_cm}")
      return pixel_map
    logger.info(f"[set_pixel_by_radius PT-MODE raw] xy={pixel_xy} radius_cm={radius_cm}")
    pixel_map[_r, _c] = value
    if radius_cm > 0:
      radius_x = self.cm2index(radius_cm, 'x')
      radius_y = self.cm2index(radius_cm, 'y')
      _ph, _pw = pm_shape[0], pm_shape[1]
      min_x = max(0, _r - radius_x)
      max_x = min(_ph, _r + radius_x + 1)
      min_y = max(0, _c - radius_y)
      max_y = min(_pw, _c + radius_y + 1)
      pixel_map[min_x:max_x, min_y:max_y] = value
    return pixel_map

  def world_offset(self, obj, dx_m=0.0, dy_m=0.0, dz_m=0.0):
    """Return a 3-tuple `_position_world`-style coord offset from `obj` by
    (dx_m, dy_m, dz_m) in WORLD frame metres. Result can be passed to
    `set_pixel_by_radius` as the second arg — it goes through POINT-WORLD
    mode (correct map_h × map_w grid conversion via workspace bounds).

    Use this instead of pixel-frame arithmetic on `obj.position`:
        # OLD (broken on non-100×100 grids; silent corner cell on dummy obj):
        x = obj.position[0] + cm2index(15, 'x')
        y = obj.position[1]
        set_pixel_by_radius(map, [x, y], radius_cm=30)   # POINT-RAW mode
        # NEW (frame-correct; falls back gracefully on missing/dummy obj):
        pt = world_offset(obj, dx_m=0.15, dy_m=0)
        set_pixel_by_radius(map, pt, radius_cm=30)       # POINT-WORLD mode

    Convention (world frame, robocasa kitchen):
        +dx_m → +x world (configurable per layout)
        +dy_m → +y world
    Returns a wrapper dict with `_position_world` so set_pixel_by_radius
    detects POINT-WORLD mode automatically.
    """
    try:
      base = np.asarray(obj._position_world if hasattr(obj, '_position_world')
                        else obj.get('_position_world', [0., 0., 0.]))[:3].astype(float)
    except Exception:
      base = np.zeros(3, dtype=float)
    out = base + np.array([float(dx_m), float(dy_m), float(dz_m)], dtype=float)
    return Observation({
      '_position_world': out,
      'name': f"world_offset({getattr(obj, 'name', '?')})",
    })

  def get_empty_affordance_map(self, task='navigation'):
    return self._get_default_voxel_map('target', task=task)()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)

  def get_empty_avoidance_map(self, task='navigation'):
    return self._get_default_voxel_map('obstacle', task=task)()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_rotation_map(self, task='navigation'):
    return self._get_default_voxel_map('rotation', task=task)()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_velocity_map(self, task='navigation'):
    return self._get_default_voxel_map('velocity', task=task)()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def reset_to_default_pose(self):
     self._env.reset_to_default_pose()
  
  # ======================================================
  # == helper functions
  # ======================================================
  def _world_to_pos_coords(self, world_xyz):
    """Position coords usable as direct array index for the LLM-set maps.

    For NAVIGATION (rectangular pixel grid map_h × map_w): returns
    (row=y_idx, col=x_idx, z_idx) using (map_h, map_w) so that
    `pixel_map[pos[0], pos[1]] = 1` lands at the correct cell.

    For MANIPULATION (cubic voxel grid map_size³): returns voxel coords
    via _world_to_voxel (legacy behaviour).
    """
    is_nav = bool(getattr(self._env, 'navigate_task', False))
    if not is_nav:
        return self._world_to_voxel(world_xyz)
    w = np.asarray(world_xyz, dtype=np.float32)
    ws_min = self._env.workspace_bounds_min.astype(np.float32)
    ws_max = self._env.workspace_bounds_max.astype(np.float32)
    rng = ws_max - ws_min
    col = int(round((w[0] - ws_min[0]) / max(rng[0], 1e-9) * (self._map_w - 1)))
    row = int(round((w[1] - ws_min[1]) / max(rng[1], 1e-9) * (self._map_h - 1)))
    z   = int(round((w[2] - ws_min[2]) / max(rng[2], 1e-9) * (self._map_size - 1)))
    row = max(0, min(self._map_h - 1, row))
    col = max(0, min(self._map_w - 1, col))
    # Navigation `pixel_map[i, j]` semantics → i=row, j=col. So return
    # (row, col, z) so direct LLM indexing `m[pos[0], pos[1]] = 1` lands
    # at correct cell.
    return np.array([row, col, z], dtype=np.int32)

  def _world_to_voxel(self, world_xyz):
    _world_xyz = world_xyz.astype(np.float32)
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    voxel_xyz = pc2voxel(_world_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return voxel_xyz

  def _voxel_to_world(self, voxel_xyz):
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    world_xyz = voxel2pc(voxel_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return world_xyz

  def _points_to_voxel_map(self, points):
    """convert points in world frame to voxel frame, voxelize, and return the voxelized points"""
    _points = points.astype(np.float32)
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    return pc2voxel_map(_points, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)

  def _get_voxel_center(self, voxel_map):
    """calculte the center of the voxel map where value is 1"""
    voxel_center = np.array(np.where(voxel_map == 1)).mean(axis=1)
    return voxel_center

  def _get_scene_collision_voxel_map(self):
    collision_points_world, _ = self._env.get_scene_3d_obs(ignore_robot=True)
    collision_voxel = self._points_to_voxel_map(collision_points_world)
    return collision_voxel

  def _get_scene_collision_pixel_map(self):
    """Build scene_collision from KITCHEN FIXTURE GEOM AABBs (rectangle
    projection), not from camera point cloud.

    Why not point cloud:
      - Patchy: cameras see only exposed surfaces of tall fixtures, so the
        interior footprint is blank → produces salt-pepper noise & holes.
      - Adding binary_closing/opening masks the symptom, doesn't fix it.

    Why rectangle (not geom_rbound disk):
      - geom_rbound = enclosing-sphere radius. For walls (long thin geoms)
        rbound ≈ length/2, projecting them as huge disks that blanket the
        kitchen (this was the bug in the previous fixture-AABB attempt that
        caused the revert to point-cloud).
      - geom_size = actual half-extents along each axis. For a box geom
        the rectangle [p±size_x, p±size_y] is the true footprint.

    Falls back to point-cloud + closing if env unavailable.
    """
    if hasattr(self, '_env') and self._env is not None:
      mask = self._get_fixture_floor_footprint(self._map_h, self._map_w)
      if mask is not None:
        return mask.astype(np.float64)
    # Fallback: legacy point-cloud + morphological closing.
    collision_points_world, _ = self._env.get_scene_3d_obs(ignore_robot=True)
    collision_pixel = self._points_to_pixel_map(collision_points_world)
    from scipy.ndimage import binary_closing
    return binary_closing(collision_pixel > 0, iterations=3).astype(np.float64)

  def _points_to_pixel_map(self, points):
    """Project world points → (map_h, map_w) pixel map (rectangular grid)."""
    _points = points.astype(np.float32)
    _pixel_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)[:2]
    _pixel_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)[:2]
    return pc2pixel_map(_points, _pixel_bounds_robot_min, _pixel_bounds_robot_max,
                        self._map_h, self._map_w)


  def _get_default_voxel_map(self, type='target', task='navigation'):
    """returns default voxel map (defaults to current state)"""
    def fn_wrapper():
      # Rectangular grid: (map_h, map_w) for isotropic 5cm cells per layout.
      if type == 'target':
        voxel_map = np.zeros((self._map_h, self._map_w))
      elif type == 'obstacle':
        voxel_map = np.zeros((self._map_h, self._map_w))
      elif type == 'velocity':
        voxel_map = np.ones((self._map_h, self._map_w))
      elif type == 'rotation':
        # NaN sentinel = "no rotation requirement at this cell".
        # Controller treats NaN waypoints as "keep current yaw" so the
        # holonomic base sidesteps freely without forced rotation. LMP
        # only writes scalar yaw to cells that need explicit alignment
        # (e.g. set_pixel_by_radius around a face-toward target).
        voxel_map = np.full((self._map_h, self._map_w), np.nan)
      else:
        raise ValueError('Unknown voxel map type: {}'.format(type))
      voxel_map = VoxelIndexingWrapper(voxel_map)
      return voxel_map
    return fn_wrapper
  
  def _path2traj_navigation(self, path, avoidance_map, rotation_map, velocity_map):
    """Convert pixel path to navigation trajectory with rotation and velocity."""
    if velocity_map is None:
      velocity_map = np.ones((self._map_size, self._map_size))
    traj = []
    cur_xy = self._env.env._get_observations()['robot0_base_pos'][:2]
    # initial_filtering drops path waypoints within 0.4m of the robot. For
    # pure_pursuit this DELETES the A* path's safe wall-exit route → the
    # controller cuts straight across it into walls. pure_pursuit projects
    # onto the path so near-robot waypoints are harmless → keep full path.
    initial_filtering = (self._nav_mode != 'pure_pursuit')
    # Rectangular-grid world conversion (scalar map_size was wrong for
    # non-square workspaces — produced traj_world starting at a totally
    # different position from path_pixel start, making (e) Cost panel
    # red trajectory and (f) Trajectory pink path show inconsistent
    # routes).
    _ws_min = np.asarray(self._env.workspace_bounds_min[:2], dtype=np.float32)
    _ws_max = np.asarray(self._env.workspace_bounds_max[:2], dtype=np.float32)
    _rng = _ws_max - _ws_min
    _mh = max(int(self._map_h) - 1, 1)
    _mw = max(int(self._map_w) - 1, 1)
    for path_idx in path:
      # path_idx = (row, col). row indexes y, col indexes x.
      world_x = float(_ws_min[0] + path_idx[1] / _mw * _rng[0])
      world_y = float(_ws_min[1] + path_idx[0] / _mh * _rng[1])
      world_xy = np.array([world_x, world_y])
      if initial_filtering:
          if np.linalg.norm(world_xy - cur_xy) < 0.4:
            continue
          else:
            initial_filtering = False
      voxel_xy = np.round(path_idx).astype(int)
      rotation = rotation_map[voxel_xy[0], voxel_xy[1]]
      velocity = velocity_map[voxel_xy[0], voxel_xy[1]]
      traj.append((world_xy, rotation, velocity))
    # v26: revert to last 1 wp injection (v20 baseline). Last 3 wps caused
    # premature target_ori commands at intermediate wps, distorting path.
    LAST_N_WITH_TARGET_YAW = 1
    if len(traj) > 0:
      dst_is_human = getattr(self._env.env, 'dst_is_human', False)
      target_yaw = None
      if dst_is_human:
        human_pos = getattr(self._env.env, 'target_pos', None)
        if human_pos is not None:
          last_wp = traj[-1]
          dir_to_human = np.array(human_pos[:2]) - np.array(last_wp[0])
          dist = np.linalg.norm(dir_to_human)
          if dist > 0.1:
            # R(π-yaw): body faces (dx,dy) when real_yaw = atan2(dy, -dx)
            target_yaw = float(np.arctan2(dir_to_human[1], -dir_to_human[0]))
          elif len(traj) >= 2:
            prev_wp = traj[-2][0]
            dir_approach = np.array(last_wp[0]) - np.array(prev_wp)
            target_yaw = float(np.arctan2(dir_approach[1], -dir_approach[0]))
      else:
        target_ori = getattr(self._env.env, 'target_ori', None)
        if target_ori is not None:
          target_yaw = float(target_ori[2])
      if target_yaw is not None:
        n_inject = min(LAST_N_WITH_TARGET_YAW, len(traj))
        for offset in range(1, n_inject + 1):
          wp = traj[-offset]
          traj[-offset] = (wp[0], target_yaw, wp[2])
        logger.debug(f'[{get_clock_time()}] injected target_yaw={np.degrees(target_yaw):.1f}deg into last {n_inject} waypoints')
    return traj
  
  def _navigate_to_trajectory(self, waypoint, to_waypoint, kp=10, lookahead_wp=None):
    goal_xy, goal_yaw, goal_vel = waypoint
    to_goal_xy = to_waypoint[0]
    direction_vector = to_goal_xy - goal_xy
    cur_xy = self._env.env._get_observations()['robot0_base_pos']
    _q = self._env.env._get_observations()['robot0_base_quat']  # robosuite xyzw
    cur_yaw = quat2euler([_q[3], _q[0], _q[1], _q[2]])[2]       # → wxyz, [2]=yaw_Z

    # nav_mode=translate_only: rotate-first toward NEXT waypoint, then
    # forward-only translation toward the SAME next waypoint. Avoids
    # holonomic v_y lateral drift. Pattern lifted from b7fab21 rotate-first.
    # IMPORTANT: heading target = translation target. Using a lookahead wp
    # for heading caused body-forward to point away from the actual next wp
    # (Person G smoke v2 drifted laterally 0.55m). Smoke-validated tuning:
    # omega_max=0.3 keeps lateral drift <4mm/step; v_x=1.0 perfectly forward.
    if self._nav_mode == 'translate_only':
      next_xy = np.asarray(to_waypoint[0])
      # R(π-yaw): body faces (dx,dy) at real_yaw = atan2(dy, -dx)
      target_yaw_face = float(np.arctan2(next_xy[1] - cur_xy[1], -(next_xy[0] - cur_xy[0])))
      delta_yaw_face = (target_yaw_face - cur_yaw + np.pi) % (2 * np.pi) - np.pi
      ROTATE_ALIGN_TOL = np.deg2rad(10)
      ROTATE_KP = 3.0
      ROTATE_OMEGA_MAX = 0.3
      TRANSLATE_KP = 5.0
      TRANSLATE_VX_MAX = 1.0
      action = np.zeros(3)
      if abs(delta_yaw_face) > ROTATE_ALIGN_TOL:
        action[2] = float(np.clip(delta_yaw_face * ROTATE_KP,
                                  -ROTATE_OMEGA_MAX, ROTATE_OMEGA_MAX))
      else:
        dist_to_wp = float(np.hypot(next_xy[0] - cur_xy[0], next_xy[1] - cur_xy[1]))
        action[0] = float(np.clip(dist_to_wp * TRANSLATE_KP, 0.0, TRANSLATE_VX_MAX))
      return action, delta_yaw_face

    # Lookahead smoothing — drive toward a point L metres ahead of goal_xy.
    # L=0.15 stays below the 25cm wp spacing (target_spacing=5) so wps are
    # never skipped, while giving enough forward bias for smooth motion.
    seg_len = np.linalg.norm(direction_vector) + 1e-8
    seg_dir = direction_vector / seg_len
    L = 0.15
    target_xy = goal_xy + seg_dir * min(L, seg_len)
    is_last = np.array_equal(goal_xy, target_xy)
    goal_yaw_scalar = np.asarray(goal_yaw).item() if np.asarray(goal_yaw).size == 1 else float('nan')
    # Face direction (simple rule):
    #   - intermediate wp: face the NEXT wp (wp[i+1] = to_waypoint)
    #   - last wp: face target_ori (= goal_yaw_scalar, success criterion)
    if is_last:
      goal_yaw = goal_yaw_scalar if not np.isnan(goal_yaw_scalar) else cur_yaw
    elif np.isnan(goal_yaw_scalar):
      # NEW: heading target = NEXT SEGMENT direction (to_wp → lookahead_wp),
      # so robot pre-aligns to where it will be heading AFTER current wp.
      # Falls back to (cur_pos → to_wp) if no lookahead given.
      _next_xy = np.asarray(to_waypoint[0])
      if lookahead_wp is not None:
        _lookahead_xy = np.asarray(lookahead_wp[0])
        _dxy_next = _lookahead_xy - _next_xy   # next segment direction
      else:
        _dxy_next = _next_xy - cur_xy[:2]
      if np.linalg.norm(_dxy_next) > 0.05:
        # R(π-yaw): body faces (dx,dy) at real_yaw = atan2(dy, -dx)
        goal_yaw = float(np.arctan2(_dxy_next[1], -_dxy_next[0]))
      else:
        goal_yaw = cur_yaw
    else:
      goal_yaw = goal_yaw_scalar

    dx = target_xy[0] - cur_xy[0]
    dy = target_xy[1] - cur_xy[1]
    delta_yaw = (goal_yaw - cur_yaw + np.pi) % (2 * np.pi) - np.pi

    # Controller's body→world is R(π - yaw). World→body inverse uses R(π-yaw)^T:
    #   v_x_body = -cos(yaw)·dx + sin(yaw)·dy
    #   v_y_body = -sin(yaw)·dx - cos(yaw)·dy
    # Audit-confirmed: yaw=0/180° tests showed standard R(yaw) gives 180° wrong direction.
    v_x = -dx * np.cos(cur_yaw) + dy * np.sin(cur_yaw)
    v_y = -dx * np.sin(cur_yaw) - dy * np.cos(cur_yaw)
    # Asymmetric gains: aggressive translation, gentle rotation. Both
    # tunable via env vars (CTRL_KP_ROT, CTRL_OMEGA_MAX) for sweep tests.
    # CRITICAL: omega does NOT get goal_vel scaling. When LMP velocity_map
    # outputs near-zero near obstacles ("slow to 5% within 30cm of human"),
    # if omega were gated too, rotation collapses to ~0.05°/step and robot
    # never faces the next waypoint. Translation gating by goal_vel is
    # correct (safety = slow near obstacle); rotation gating is not.
    KP_ROT = float(os.environ.get('CTRL_KP_ROT', '3.0'))
    OMEGA_MAX = float(os.environ.get('CTRL_OMEGA_MAX', '0.3'))
    # Incremental-rotate mode: cap omega to a very small value so per-step
    # drift is sub-mm, letting alignment happen gradually alongside
    # translation. Drift accumulated over the rotation is the same total,
    # but spread across the motion → robot makes path progress meanwhile.
    if os.environ.get('INCREMENTAL_ROTATE_ENABLED', '0') == '1':
      OMEGA_MAX = float(os.environ.get('INCREMENTAL_OMEGA_MAX', '0.05'))
    action = np.zeros(3)
    action[0] = v_x * goal_vel * kp
    action[1] = v_y * goal_vel * kp
    action[2] = delta_yaw * KP_ROT
    # OMEGA_GATING=1: when |delta_yaw| > threshold, send rotation-only
    # ([0, 0, omega]); else translation-only ([v_x, v_y, 0]). Avoids
    # omega+v_x interference (data: omega kills v_x to 30%).
    if os.environ.get('OMEGA_GATING', '0') == '1':
      _gate_rad = float(os.environ.get('OMEGA_GATING_THRESH_RAD', '0.087'))   # ~5°
      if abs(delta_yaw) > _gate_rad:
        action[0] = 0.0
        action[1] = 0.0
        # keep action[2] = delta_yaw * KP_ROT
      else:
        action[2] = 0.0
    # v26: rotate-first removed (v20 baseline). With NaN→cur_yaw default,
    # intermediate wps have delta_yaw=0 so rotate-first wouldn't trigger
    # anyway. Only the last wp uses target_ori (where delta_yaw can be big),
    # but Y4 early-exit handles that with dist+yaw_threshold check.
    #
    # Y6 (v28→v29): at LAST waypoint, suppress translation when robot is
    # CLOSE to wp AND yaw not aligned, so it rotates in place without
    # drifting. Without this, action[1] (body-frame translation toward wp)
    # keeps firing while yaw rotates → body-y direction shifts in world
    # frame → robot spirals away from goal area.
    # v29: threshold raised 0.5→1.0 — L8 v28 case showed robot at dist=0.53m
    # (just outside 0.5m) still triggered drift over 300 steps. 1.0m gives
    # the rotation enough headroom before robot enters jam-prone region.
    Y6_TRANSLATION_SUPPRESS_DIST_M = 1.0
    if is_last:
      _dxy_now = np.hypot(target_xy[0] - cur_xy[0], target_xy[1] - cur_xy[1])
      if _dxy_now <= Y6_TRANSLATION_SUPPRESS_DIST_M and abs(delta_yaw) > self._yaw_threshold:
        action[0] = 0.0
        action[1] = 0.0
    # Asymmetric clip: translation can saturate at ±1.0 (25mm/step max),
    # but rotation capped at ±OMEGA_MAX to limit per-step lateral drift.
    action[:2] = np.clip(action[:2], -1.0, 1.0)
    action[2] = float(np.clip(action[2], -OMEGA_MAX, OMEGA_MAX))
    return action, delta_yaw

  def _preprocess_avoidance_voxel_map(self, avoidance_map, affordance_map, movable_obs):
    # collision avoidance
    scene_collision_map = self._get_scene_collision_voxel_map()
    # anywhere within 15/100 indices of the target is ignored (to guarantee that we can reach the target)
    ignore_mask = distance_transform_edt(1 - affordance_map)
    scene_collision_map[ignore_mask < int(0.2 * self._map_size)] = 0
    # anywhere within 15/100 indices of the start is ignored
    try:
      ignore_mask = distance_transform_edt(1 - movable_obs['occupancy_map'])
      scene_collision_map[ignore_mask < int(0.1 * self._map_size)] = 0
    except KeyError:
      start_pos = movable_obs['position']
      ignore_mask = np.ones_like(avoidance_map)
      ignore_mask[start_pos[0] - int(0.1 * self._map_size):start_pos[0] + int(0.1 * self._map_size),
                  start_pos[1] - int(0.1 * self._map_size):start_pos[1] + int(0.1 * self._map_size),
                  start_pos[2] - int(0.1 * self._map_size):start_pos[2] + int(0.1 * self._map_size)] = 0
      scene_collision_map *= ignore_mask
    avoidance_map += scene_collision_map
    avoidance_map = np.clip(avoidance_map, 0, 1)
    return avoidance_map

  def _preprocess_avoidance_pixel_map(self, avoidance_map, affordance_map, movable_obs,
                                      robot_radius=0.50, r_scene_m=0.55, r_llm_m=0.32):
    scene_collision_map = self._get_scene_collision_pixel_map()
    # Raw fixture obstacles BEFORE any target/start clearing. The robot can
    # spawn closer than robot_radius to a fixture (e.g. fridge ~0.34m away);
    # the robot-footprint mask (step E) would then erase that fixture from
    # the avoidance map → A* routed a path through it → robot wedged. Keep a
    # snapshot so step E never clears a genuine fixture cell.
    scene_obstacle_raw = scene_collision_map > 0
    H, W = avoidance_map.shape
    # (A) Radii expressed in meters and converted to pixels via the actual cell
    # resolution — robust to workspace size changes.
    # r_scene_m ≥ 0.5m goal-success threshold so the robot can reach the goal.
    xy = self._compute_pixel_resolution()
    cell_m = float(xy.min())
    r_scene = int(np.ceil(r_scene_m / cell_m))
    r_llm   = int(np.ceil(r_llm_m / cell_m))
    # (C) Use full affordance footprint (not just argmax) as the target region —
    # fall back to argmax if affordance is empty so behavior degrades gracefully.
    target_mask = affordance_map > 0
    if not target_mask.any():
        ti = np.unravel_index(np.argmax(affordance_map), affordance_map.shape)
        target_mask = np.zeros_like(affordance_map, dtype=bool)
        target_mask[ti] = True
    # (B) Circular clear via Euclidean distance transform — isotropic, no axial bias.
    # avoidance_map may be a VoxelIndexingWrapper; reach the underlying ndarray
    # for boolean-mask indexing (the wrapper's __setitem__ chokes on bool masks).
    av_arr = avoidance_map.array if hasattr(avoidance_map, 'array') else avoidance_map
    target_dist = distance_transform_edt(~target_mask)
    scene_collision_map[target_dist <= r_scene] = 0
    av_arr[target_dist <= r_llm] = 0
    # Clear collision around robot start position so A* can move out.
    # NOTE: this zeroes the RAW scene_collision (real counters/walls) inside
    # the disk. The legacy radius (robot_radius/cell+1 ≈ 0.56m) wiped real
    # counters near the spawn → A* routed paths THROUGH them → robot wedged.
    # START_CLEAR_CELLS shrinks the disk to the minimum needed for A* escape
    # (default small) so genuine counters near the start stay in the map.
    start_pos = movable_obs['position']
    sp0, sp1 = int(start_pos[0]), int(start_pos[1])
    xy = self._compute_pixel_resolution()
    _legacy_margin = int(np.ceil(robot_radius / float(xy.min()))) + 1
    # Best config (sweep 2026-05-24): START_CLEAR_CELLS=2 (small disk so
    # real counters near the start stay in the map; was legacy ~10 cells).
    margin = int(os.environ.get('START_CLEAR_CELLS', '2'))
    yy, xx = np.ogrid[:H, :W]
    start_clear = (yy - sp0)**2 + (xx - sp1)**2 <= margin**2
    scene_collision_map[start_clear] = 0
    av_arr += scene_collision_map
    np.clip(av_arr, 0, 1, out=av_arr)
    # (E) Robot footprint mask — project every robot body's geom_xpos to floor
    # and force those cells (and a 1-cell dilation buffer) to be free. The
    # camera-mask exclusion in get_scene_3d_obs handles point-cloud level, but
    # the 35cm start_clear circle alone may miss the arm/gripper footprint
    # when extended.
    robot_mask = self._get_robot_floor_footprint(H, W)
    if robot_mask is not None:
        # Never clear genuine fixtures even when they fall inside the robot
        # footprint disk — only residual non-fixture cells get freed.
        robot_mask = robot_mask & ~scene_obstacle_raw
        av_arr[robot_mask] = 0
    # (D) Force workspace boundary ring to obstacle — applied AFTER all clearing
    # logic so the border is never carved out by goal/start clear.
    border_w = max(1, int(np.ceil(0.10 / cell_m)))   # ~10cm ring
    av_arr[:border_w, :] = 1.0
    av_arr[-border_w:, :] = 1.0
    av_arr[:, :border_w] = 1.0
    av_arr[:, -border_w:] = 1.0
    return avoidance_map

  def _get_robot_floor_footprint(self, H, W):
    """Project all robot body geom positions to a 2D floor mask in the
    avoidance grid. Returns (H, W) bool array or None if env unavailable.
    Uses workspace_bounds to convert world XY → grid index. Dilates by 1
    cell so adjacent corners are also cleared.
    """
    try:
      sim = self._env.env.sim
      model = sim.model
      wmin = self._env.workspace_bounds_min[:2]
      wmax = self._env.workspace_bounds_max[:2]
    except (AttributeError, TypeError):
      return None
    if wmax[0] - wmin[0] <= 0 or wmax[1] - wmin[1] <= 0:
      return None
    # Identify robot body IDs by name pattern. Include gripper bodies (their
    # geoms can extend horizontally when arm is pointed forward and project
    # to floor cells outside the start_clear circle). Exclude eef_target
    # bodies (placed at z=-1, below floor; would land as phantom obstacle
    # near origin XY).
    robot_body_ids = set()
    for i in range(model.nbody):
      n = (model.body_id2name(i) or '').lower()
      if 'eef_target' in n:
        continue
      if any(p in n for p in ('robot0', 'mobilebase', 'gripper0', 'panda')):
        robot_body_ids.add(i)
    if not robot_body_ids:
      return None
    # Project EACH robot geom as a disk of radius geom_rbound (the geom's
    # enclosing-sphere radius — a safe upper bound for the floor footprint).
    # Previously we only marked the geom's xpos centroid + 1-cell dilation,
    # which underestimates large geoms (e.g. mobilebase0_wheeled_base
    # rbound=0.541m → ~10 cells radius vs 1 cell with the old code).
    # Pixel grid convention: H=map_h=y_cells (rows), W=map_w=x_cells (cols).
    # cell_x = world x extent / W,  cell_y = world y extent / H.
    sx_m = (wmax[0] - wmin[0]) / W   # cell width along world-x = col axis
    sy_m = (wmax[1] - wmin[1]) / H   # cell height along world-y = row axis
    cell_min_m = float(min(sx_m, sy_m))
    yy, xx = np.ogrid[:H, :W]
    mask = np.zeros((H, W), dtype=bool)
    for gid in range(model.ngeom):
      bid = int(model.geom_bodyid[gid])
      if bid not in robot_body_ids:
        continue
      p = sim.data.geom_xpos[gid]
      # Skip geoms below the floor (z<0 markers) AND elevated arm/gripper
      # geoms (z>z_max). The arm is OVERHEAD — projecting it to the floor and
      # forcing those cells free wrongly erases real counters UNDER the
      # raised arm (the robot BASE cannot go there). Only mobile-base geoms
      # (z≤~0.5m) define the true navigable floor footprint.
      _zmax = float(os.environ.get('ROBOT_FOOTPRINT_Z_MAX', '0.6'))
      if p[2] < -0.05 or p[2] > _zmax:
        continue
      # Project the geom's ACTUAL XY footprint rectangle (geom_size half-
      # extents), NOT a geom_rbound enclosing-sphere disk. rbound hugely
      # over-estimates wide/flat base geoms (rbound≈0.5m) → the robot mask
      # ballooned and erased real counters (fridge) 0.3m+ from the robot.
      gs = model.geom_size[gid]
      hx, hy = float(gs[0]), float(gs[1])
      if hx <= 0 or hy <= 0:
        continue
      gmat = sim.data.geom_xmat[gid].reshape(3, 3)
      half_x = abs(gmat[0, 0]) * hx + abs(gmat[0, 1]) * hy
      half_y = abs(gmat[1, 0]) * hx + abs(gmat[1, 1]) * hy
      x0 = max(p[0] - half_x, wmin[0]); x1 = min(p[0] + half_x, wmax[0])
      y0 = max(p[1] - half_y, wmin[1]); y1 = min(p[1] + half_y, wmax[1])
      if x1 <= x0 or y1 <= y0:
        continue
      c0 = int(np.floor((x0 - wmin[0]) / sx_m)); c1 = int(np.ceil((x1 - wmin[0]) / sx_m))
      r0 = int(np.floor((y0 - wmin[1]) / sy_m)); r1 = int(np.ceil((y1 - wmin[1]) / sy_m))
      r0 = max(0, r0); r1 = min(H, r1); c0 = max(0, c0); c1 = min(W, c1)
      if r1 <= r0 or c1 <= c0:
        continue
      mask[r0:r1, c0:c1] = True
    if not mask.any():
      return None
    return mask
  
  def _get_fixture_floor_footprint(self, H, W):
    """Project all kitchen FIXTURE geom AABBs to a 2D floor mask using
    AXIS-ALIGNED RECTANGLES (geom_size half-extents).

    This is the second iteration of fixture-AABB projection. The first
    iteration used `geom_rbound` (enclosing-sphere radius) which made walls
    project as huge disks blanketing the kitchen, forcing a revert to point
    cloud. This version uses each geom's actual XY half-extents so walls
    become thin rectangles, counters become real footprints, etc.

    For non-axis-aligned geoms the rectangle is widened by the rotated
    bounding box of the original (size_x, size_y) — conservative and exact
    for axis-aligned fixtures (which is the common case in robocasa).

    Bodies excluded: robot, floor (navigable), eef_target, world,
    standing_table. Geoms below z=-0.05 (sub-floor decals) and above
    z=1.8m (ceiling lamps) are skipped as they cannot collide with the
    mobile base.

    Returns (H, W) bool array or None if env unavailable.
    """
    try:
      sim = self._env.env.sim
      model = sim.model
      wmin = self._env.workspace_bounds_min[:2]
      wmax = self._env.workspace_bounds_max[:2]
    except (AttributeError, TypeError):
      return None
    if wmax[0] - wmin[0] <= 0 or wmax[1] - wmin[1] <= 0:
      return None

    # Note: 'standing_table' was previously excluded (manipulation-era
    # assumption that the table is a worksurface). For navigation it is
    # a real obstacle (0.88 m tall × 0.30 m radius) and must be in the
    # collision map, so we no longer exclude it.
    exclude_patterns = ('robot0', 'mobilebase', 'gripper0', 'panda',
                        'eef_target', '_target', 'world')
    exclude_body_ids = set()
    for i in range(model.nbody):
      n = (model.body_id2name(i) or '').lower()
      if any(p in n for p in exclude_patterns):
        exclude_body_ids.add(i)

    # World→grid conversion: H=map_h=y_cells (rows), W=map_w=x_cells (cols).
    # Cell size meters: x extent / W (col=x), y extent / H (row=y).
    sx_m = (wmax[0] - wmin[0]) / W   # cell width in world x = col axis
    sy_m = (wmax[1] - wmin[1]) / H   # cell height in world y = row axis
    mask = np.zeros((H, W), dtype=bool)

    for gid in range(model.ngeom):
      gname = (model.geom_id2name(gid) or '').lower()
      if 'floor' in gname:
        continue
      bid = int(model.geom_bodyid[gid])
      if bid in exclude_body_ids:
        continue
      p = sim.data.geom_xpos[gid]
      if p[2] < -0.05 or p[2] > 1.8:
        continue
      # geom_size meaning depends on geom type, but for box/mesh/cylinder
      # the first two entries are XY half-extents in the geom's local frame.
      # For sphere, all entries equal radius — also OK as half-extents.
      gs = model.geom_size[gid]
      hx, hy = float(gs[0]), float(gs[1])
      if hx <= 0 or hy <= 0:
        continue
      # Account for rotation: use rotated AABB so that diagonal orientation
      # still produces a valid (over-approximate) world-axis footprint.
      mat = sim.data.geom_xmat[gid].reshape(3, 3)
      # |R[0,0]|*hx + |R[0,1]|*hy = world-x half-extent of rotated rect
      half_x = abs(mat[0, 0]) * hx + abs(mat[0, 1]) * hy
      half_y = abs(mat[1, 0]) * hx + abs(mat[1, 1]) * hy

      x0 = p[0] - half_x
      x1 = p[0] + half_x
      y0 = p[1] - half_y
      y1 = p[1] + half_y
      # Clip to workspace bounds
      x0 = max(x0, wmin[0]); x1 = min(x1, wmax[0])
      y0 = max(y0, wmin[1]); y1 = min(y1, wmax[1])
      if x1 <= x0 or y1 <= y0:
        continue
      # World → grid (correct convention):
      #   col index = (x - wmin_x) / cell_x  (world-x → col axis, width W)
      #   row index = (y - wmin_y) / cell_y  (world-y → row axis, height H)
      c0 = int(np.floor((x0 - wmin[0]) / sx_m))
      c1 = int(np.ceil ((x1 - wmin[0]) / sx_m))
      r0 = int(np.floor((y0 - wmin[1]) / sy_m))
      r1 = int(np.ceil ((y1 - wmin[1]) / sy_m))
      r0 = max(0, r0); r1 = min(H, r1)
      c0 = max(0, c0); c1 = min(W, c1)
      if r1 <= r0 or c1 <= c0:
        continue
      mask[r0:r1, c0:c1] = True

    if not mask.any():
      return None
    return mask

  def _compute_pixel_resolution(self):
    world_xy = self._voxel_to_world(np.array([1,1,0]))[:2] - self._voxel_to_world(np.array([0,0,0]))[:2]
    return world_xy

def setup_LMP(env, general_config, debug=False, output_dir=None):
  controller_config = general_config['controller']
  planner_config = general_config['planner']
  lmp_env_config = general_config['lmp_config']['env']
  lmps_config = general_config['lmp_config']['lmps']
  env_name = general_config['env_name']
  llm_api_config = general_config.get('llm_api', {})
  nav_controller_config = general_config.get('navigation_controller', {})
  # LMP env wrapper
  lmp_env = NavigationLMPInterface(env, lmp_env_config, controller_config, planner_config, env_name=env_name, nav_controller_config=nav_controller_config, output_dir=output_dir)
  # creating APIs that the LMPs can interact with
  import time as _time
  fixed_vars = {
      'np': np,
      'euler2quat': transforms3d.euler.euler2quat,
      'quat2euler': transforms3d.euler.quat2euler,
      'qinverse': transforms3d.quaternions.qinverse,
      'qmult': transforms3d.quaternions.qmult,
      'vec2quat': _vec2quat,
      'time': _time,
  }  # external library APIs
  variable_vars = {
      k: getattr(lmp_env, k)
      for k in dir(lmp_env) if callable(getattr(lmp_env, k)) and not k.startswith("_")
  }  # our custom APIs exposed to LMPs

  # Pre-define 'movable' as the robot mobile base so LLM-generated code that omits
  # `movable = parse_query_obj('mobile_base')` still has it in scope.
  # Use DynamicObservation so detect() is called lazily (after env.load_task()).
  if 'navigation' in env_name:
      variable_vars['movable'] = DynamicObservation(lambda: lmp_env.detect('mobile_base'))

  # allow LMPs to access other LMPs
  lmp_names = [name for name in lmps_config.keys() if not name in ['composer', 'planner', 'config'] and lmps_config[name] is not None]
  low_level_lmps = {
      k: LMP(k, lmps_config[k], fixed_vars, variable_vars, debug, env_name, llm_api_config=llm_api_config)
      for k in lmp_names
  }
  variable_vars.update(low_level_lmps)

  # Wrap parse_query_obj to return a safe fallback Observation when object not found,
  # preventing 'NoneType' crashes in LLM-generated map code.
  # Must be placed AFTER variable_vars.update(low_level_lmps) so the LMP version is wrapped.
  # Fallback carries every key a real detect() Observation has so LLM code paths
  # like `obj.occupancy_map` / `obj._point_cloud_world` don't KeyError on miss.
  _M = lmp_env._map_size
  _SAFE_FALLBACK_OBS = Observation({
      'name':                '_fallback',
      'position':            np.array([0.0, 0.0, 0.0]),
      'normal':              np.array([0.0, 0.0, 1.0]),
      'aabb':                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
      'occupancy_map':       np.zeros((_M, _M, _M), dtype=np.float32),
      '_position_world':     np.array([0.0, 0.0, 0.0]),
      '_point_cloud_world':  np.zeros((1, 3), dtype=np.float32),
  })
  _orig_parse_query_obj = variable_vars.get('parse_query_obj', lmp_env.detect)
  def _safe_parse_query_obj(query):
      try:
          result = _orig_parse_query_obj(query)
          if result is None:
              import logging
              logging.getLogger(__name__).warning(
                  f'[_safe_parse_query_obj] query={query!r} → None, using [0,0,0] fallback'
              )
              return _SAFE_FALLBACK_OBS
          return result
      except Exception as _e:
          import logging, traceback as _tb
          logging.getLogger(__name__).warning(
              f'[_safe_parse_query_obj] query={query!r} raised {type(_e).__name__}: {_e} '
              f'→ using [0,0,0] fallback'
          )
          logging.getLogger(__name__).debug(f'[_safe_parse_query_obj] traceback:\n{_tb.format_exc()}')
          return _SAFE_FALLBACK_OBS
  variable_vars['parse_query_obj'] = _safe_parse_query_obj

  # creating the LMP for skill-level composition
  composer = LMP(
      'composer', lmps_config['composer'], fixed_vars, variable_vars, debug, env_name, llm_api_config=llm_api_config
  )
  variable_vars['composer'] = composer

  # creating the LMP that deals w/ high-level language commands
  task_planner = LMP(
      'planner', lmps_config['planner'], fixed_vars, variable_vars, debug, env_name, llm_api_config=llm_api_config
  )

  lmps = {
      'plan_ui': task_planner,
      'composer_ui': composer,
  }
  lmps.update(low_level_lmps)

  return lmps, lmp_env


# ======================================================
# jit-ready functions (for faster replanning time, need to install numba and add "@njit")
# ======================================================
def pc2voxel(pc, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """voxelize a point cloud"""
  pc = pc.astype(np.float32)
  # make sure the point is within the voxel bounds
  pc = np.clip(pc, voxel_bounds_robot_min, voxel_bounds_robot_max)
  # voxelize
  voxels = (pc - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxels)
  voxels = np.round(voxels, 0, _out).astype(np.int32)
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  return voxels

def voxel2pc(voxels, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """de-voxelize a voxel"""
  # check voxel coordinates are non-negative
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  voxels = voxels.astype(np.float32)
  # de-voxelize
  pc = voxels / (map_size - 1) * (voxel_bounds_robot_max - voxel_bounds_robot_min) + voxel_bounds_robot_min
  return pc

def pc2voxel_map(points, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """given point cloud, create a fixed size voxel map, and fill in the voxels"""
  points = points.astype(np.float32)
  voxel_bounds_robot_min = voxel_bounds_robot_min.astype(np.float32)
  voxel_bounds_robot_max = voxel_bounds_robot_max.astype(np.float32)
  # make sure the point is within the voxel bounds
  points = np.clip(points, voxel_bounds_robot_min, voxel_bounds_robot_max)
  # voxelize
  voxel_xyz = (points - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxel_xyz)
  points_vox = np.round(voxel_xyz, 0, _out).astype(np.int32)
  voxel_map = np.zeros((map_size, map_size, map_size))
  for i in range(points_vox.shape[0]):
      voxel_map[points_vox[i, 0], points_vox[i, 1], points_vox[i, 2]] = 1
  return voxel_map

def pc2pixel_map(points, pixel_bounds_robot_min, pixel_bounds_robot_max,
                 map_h, map_w=None, z_min=None, z_max=None):
  """Project world point cloud onto a (map_h, map_w) pixel map.

  Convention (consistent with planner / visualization):
      pixel_map[row, col] where row indexes WORLD-Y, col indexes WORLD-X.

  Earlier versions used a single `map_size` and indexed
  `pixel_map[x_idx, y_idx]` which silently swapped axes — visible as cat
  marker landing in the wrong corner for square maps and ~near-correct
  for rectangular ones (coincidence in extent ratio)."""
  if map_w is None:
      map_w = map_h
  points = points.astype(np.float32)
  z_min = points[:, 2].min() + 0.05 if z_min is None else z_min
  logger.debug(f"z min: {z_min}")
  points = points[points[:, 2] >= z_min]
  if z_max is not None:
      points = points[points[:, 2] <= z_max]
  if len(points) == 0:
      return np.zeros((map_h, map_w))

  points_xy = points[:, :2]
  pixel_bounds_robot_min = pixel_bounds_robot_min.astype(np.float32)[:2]
  pixel_bounds_robot_max = pixel_bounds_robot_max.astype(np.float32)[:2]
  points_xy = np.clip(points_xy, pixel_bounds_robot_min, pixel_bounds_robot_max)
  rng = pixel_bounds_robot_max - pixel_bounds_robot_min
  cols = ((points_xy[:, 0] - pixel_bounds_robot_min[0]) / rng[0] * (map_w - 1)).round().astype(np.int32)
  rows = ((points_xy[:, 1] - pixel_bounds_robot_min[1]) / rng[1] * (map_h - 1)).round().astype(np.int32)
  cols = np.clip(cols, 0, map_w - 1)
  rows = np.clip(rows, 0, map_h - 1)
  pixel_map = np.zeros((map_h, map_w))
  pixel_map[rows, cols] = 1
  return pixel_map