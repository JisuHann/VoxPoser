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
    # When True, parse_query_obj() / detect() auto-resolve fixture queries
    # (coffee_machine, sink, stove, ...) to fixture.pos instead of the
    # visible point-cloud centroid. Default True — flip to False to
    # restore legacy centroid behaviour.
    self._use_fixture_pos = bool(self._cfg.get('use_fixture_pos', True))

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

    Movable obstacles (cat, dog, person, ...) are not in the fixtures
    registry so they always fall back to visible centroid — which is
    what we want (avoidance halo around the actual mesh).
    """
    print("object name:", obj_name)
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
      return Observation(obs_dict)
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
        _affordance_map = affordance_map()
        _avoidance_map = avoidance_map()
        _rotation_map = rotation_map()
        _velocity_map = velocity_map()
        # Defensive fallback: LMP-generated get_*_map sometimes omits `ret_val =`
        # at the end → returns None → downstream crashes ('NoneType' subscriptable).
        # Replace any None with the corresponding default voxel map.
        if _rotation_map is None:
            _rotation_map = self._get_default_voxel_map('rotation', task='navigation')()
        if _velocity_map is None:
            _velocity_map = self._get_default_voxel_map('velocity', task='navigation')()
        if _affordance_map is None:
            _affordance_map = self._get_default_voxel_map('target', task='navigation')()
        if _avoidance_map is None:
            _avoidance_map = self._get_default_voxel_map('obstacle', task='navigation')()
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
          _robot_radius_m = 0.25  # safe default if geom_rbound lookup fails
          try:
            _model = self._env.env.sim.model
            _max_rb = 0.0
            for _gid in range(_model.ngeom):
              _bid = int(_model.geom_bodyid[_gid])
              _bn = (_model.body_id2name(_bid) or '').lower()
              # Only consider the wheeled mobile base — arm/gripper rbounds are
              # irrelevant for floor inflation (z >> floor) and would inflate
              # the planner footprint with non-floor geometry.
              if 'mobilebase' in _bn and 'wheel' in _bn:
                _rb = float(_model.geom_rbound[_gid])
                if _rb > _max_rb:
                  _max_rb = _rb
            if _max_rb > 0:
              _robot_radius_m = _max_rb / 2.0
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

        # v30: rotate-first phase before main waypoint loop.
        # If the path direction at the start is far from where the robot is
        # currently facing, rotating while translating (v26 default) makes the
        # body-frame action curve in world frame and the robot drifts away
        # from the path. So we rotate in place first — translation is
        # suppressed until cur_yaw aligns with the first segment's tangent
        # (within ROTATE_FIRST_TOL_RAD), or we hit ROTATE_FIRST_MAX_STEPS.
        # Opt-in via env var ROTATE_FIRST_ENABLED=1 (default off — initial
        # test on Cat Route A L2 produced 12.4 m goal divergence; still gated
        # while a safer trigger threshold is verified).
        _rotate_first_enabled = os.environ.get('ROTATE_FIRST_ENABLED', '0') == '1'
        if _rotate_first_enabled and len(traj_world) >= 2:
          first_dxy = np.asarray(traj_world[1][0]) - np.asarray(traj_world[0][0])
          if np.linalg.norm(first_dxy) > 0.05:
            first_tangent_yaw = float(np.arctan2(first_dxy[1], first_dxy[0]))
            _q = self._env.env._get_observations()['robot0_base_quat']
            cur_yaw0 = quat2euler([_q[3], _q[0], _q[1], _q[2]])[2]
            init_delta = (first_tangent_yaw - cur_yaw0 + np.pi) % (2 * np.pi) - np.pi
            ROTATE_FIRST_TRIGGER_RAD = np.deg2rad(60)
            ROTATE_FIRST_TOL_RAD = np.deg2rad(15)
            ROTATE_FIRST_MAX_STEPS = 30
            if abs(init_delta) > ROTATE_FIRST_TRIGGER_RAD:
              logger.info(f'[{get_clock_time()}] rotate-first phase: '
                          f'init delta_yaw={np.degrees(init_delta):+.1f}deg '
                          f'(> {np.degrees(ROTATE_FIRST_TRIGGER_RAD):.0f}deg trigger)')
              for _rf_step in range(ROTATE_FIRST_MAX_STEPS):
                _q = self._env.env._get_observations()['robot0_base_quat']
                _cy = quat2euler([_q[3], _q[0], _q[1], _q[2]])[2]
                _d = (first_tangent_yaw - _cy + np.pi) % (2 * np.pi) - np.pi
                if abs(_d) <= ROTATE_FIRST_TOL_RAD:
                  logger.info(f'[{get_clock_time()}] rotate-first done in '
                              f'{_rf_step + 1} steps (remaining delta='
                              f'{np.degrees(_d):+.1f}deg)')
                  break
                rotate_action = np.array([0.0, 0.0, np.clip(_d * 10.0, -1.0, 1.0)])
                self._env.apply_navigation_action(rotate_action)
              else:
                logger.warning(f'[{get_clock_time()}] rotate-first hit step '
                               f'limit ({ROTATE_FIRST_MAX_STEPS}); proceeding')

        for i, waypoint in tqdm(enumerate(traj_world), total=len(traj_world), desc='REACHED waypoint'):
          waypoint_reach = False
          is_last = (i == len(traj_world) - 1)
          dist_threshold = self._dist_threshold
          wp_step = 0
          while not waypoint_reach:
            if is_last:
              traj_action, dist_to_yaw = self._navigate_to_trajectory(traj_world[i], traj_world[i])
            else:
              traj_action, dist_to_yaw = self._navigate_to_trajectory(traj_world[i], traj_world[i+1])
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
      #   1. obstacle_name == "human" → _get_person_pos() (person is body)
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
        if _obstacle_name == "human" and hasattr(self._env, "_get_person_pos"):
          _p = self._env._get_person_pos()
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
  
  def set_pixel_by_radius(self, pixel_map, pixel_xy_or_obj, radius_cm=0, value=1):
    """Set `value` over a region of `pixel_map`. Two modes:

    Object mode (preferred for fixtures and grouped objects):
        pixel_xy_or_obj is an Observation dict with `occupancy_map`. The
        actual obstacle geometry is dilated by `radius_cm` cells, so the
        avoidance halo follows the real mesh — not a centroid disk that
        collapses to floor-center for perimeter-distributed obstacles
        (counter, kitchen group, etc.).

    Point mode (legacy):
        pixel_xy_or_obj is [x, y] (or [x, y, z]). Sets a square of size
        2·radius_cm centred at that point.
    """
    if pixel_map is None or pixel_xy_or_obj is None:
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
      # radius dilation operates on a single continuous mask. Closing with
      # a 25cm structuring element merges parts that are within ~25cm of
      # each other; far-separated components (multiple distinct fixtures
      # mapped to the same name) remain separate.
      from scipy.ndimage import binary_closing, distance_transform_edt
      _cell_m = float(self._resolution[0])
      _gap_cells = max(1, int(round(0.25 / _cell_m)))
      occ_pre = occ
      try:
        occ = binary_closing(occ, iterations=_gap_cells)
      except Exception:
        occ = occ_pre
      if radius_cm > 0:
        radius_cells = max(1, int(round(radius_cm / (_cell_m * 100))))
        halo = distance_transform_edt(~occ) <= radius_cells
      else:
        halo = occ
      target = pixel_map.array if hasattr(pixel_map, 'array') else pixel_map
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
      logger.info(f"[set_pixel_by_radius PT-MODE raw] skipped (None coord) xy={pixel_xy} radius_cm={radius_cm}")
      return pixel_map
    try:
      _r, _c = int(pixel_xy[0]), int(pixel_xy[1])
    except (TypeError, ValueError):
      logger.info(f"[set_pixel_by_radius PT-MODE raw] skipped (non-numeric) xy={pixel_xy} radius_cm={radius_cm}")
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
    initial_filtering = True
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
        person_pos = getattr(self._env.env, 'target_pos', None)
        if person_pos is not None:
          last_wp = traj[-1]
          dir_to_person = np.array(person_pos[:2]) - np.array(last_wp[0])
          dist = np.linalg.norm(dir_to_person)
          if dist > 0.1:
            target_yaw = float(np.arctan2(dir_to_person[1], dir_to_person[0]))
          elif len(traj) >= 2:
            prev_wp = traj[-2][0]
            dir_approach = np.array(last_wp[0]) - np.array(prev_wp)
            target_yaw = float(np.arctan2(dir_approach[1], dir_approach[0]))
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
  
  def _navigate_to_trajectory(self, waypoint, to_waypoint, kp=10):
    goal_xy, goal_yaw, goal_vel = waypoint
    to_goal_xy = to_waypoint[0]
    direction_vector = to_goal_xy - goal_xy
    cur_xy = self._env.env._get_observations()['robot0_base_pos']
    _q = self._env.env._get_observations()['robot0_base_quat']  # robosuite xyzw
    cur_yaw = quat2euler([_q[3], _q[0], _q[1], _q[2]])[2]       # → wxyz, [2]=yaw_Z

    # Lookahead smoothing — drive toward a point L metres ahead of goal_xy.
    # v26: revert to 0.3 (v20 baseline). v25 L=0.1 caused undershoot.
    seg_len = np.linalg.norm(direction_vector) + 1e-8
    seg_dir = direction_vector / seg_len
    L = 0.3
    target_xy = goal_xy + seg_dir * min(L, seg_len)
    is_last = np.array_equal(goal_xy, target_xy)
    goal_yaw_scalar = np.asarray(goal_yaw).item() if np.asarray(goal_yaw).size == 1 else float('nan')
    # v26: revert NaN→cur_yaw (v20 baseline that succeeded for L0/L6).
    # NaN→seg_dir caused intermediate-wp drift across v21-v25.
    #   1. Last wp + explicit yaw  → align to that target_ori (success criterion)
    #   2. LMP-set scalar yaw      → use it (face-toward intent at this cell)
    #   3. NaN sentinel (default)  → keep current yaw (no rotation forced)
    if is_last:
      goal_yaw = goal_yaw_scalar if not np.isnan(goal_yaw_scalar) else cur_yaw
    elif np.isnan(goal_yaw_scalar):
      goal_yaw = cur_yaw
    else:
      goal_yaw = goal_yaw_scalar

    # Drive toward the lookahead point (target_xy) instead of exact goal_xy
    # for smoother motion. The PandaOmron base is holonomic so translation
    # and rotation can happen simultaneously — there is no "facing" required
    # before moving. Previously we cosine-damped translation when off-yaw
    # (move_factor = max(0, cos(delta_yaw))) which forced the robot to
    # rotate-only when delta_yaw > 90°, exhausting the per-waypoint step
    # budget on rotation alone (verified bug: L3 wp[0] needed 90° rotation
    # → robot only rotated, never translated, then moved on to wp[1] with
    # same problem → wandering and never reaching goal).
    #
    # Fix: drop move_factor entirely. Body-frame v_x, v_y already encode
    # the correct "go-this-way-while-also-rotating" command for holonomic
    # base. The action.clip([-1, 1]) at the end caps total velocity safely.
    dx = target_xy[0] - cur_xy[0]
    dy = target_xy[1] - cur_xy[1]
    delta_yaw = (goal_yaw - cur_yaw + np.pi) % (2 * np.pi) - np.pi

    v_x = dx * np.cos(cur_yaw) + dy * np.sin(cur_yaw)
    v_y = -dx * np.sin(cur_yaw) + dy * np.cos(cur_yaw)
    action = np.zeros(3)
    action[0] = v_x * goal_vel * kp
    action[1] = v_y * goal_vel * kp
    # action[2] sign: clean omega test confirmed action[2]=+1 → CCW (yaw
    # increases via shortest signed angle). Earlier "−delta" flip was a
    # misread caused by yaw wrapping at ±π. The original convention is
    # correct: positive delta (need CCW) → positive action[2].
    action[2] = delta_yaw * goal_vel * kp
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
    action = np.clip(action, -1.0, 1.0)
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
                                      robot_radius=0.35, r_scene_m=0.55, r_llm_m=0.32):
    scene_collision_map = self._get_scene_collision_pixel_map()
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
    # Clear collision around robot start position so it can move out.
    # (B) Circular clear here too.
    start_pos = movable_obs['position']
    sp0, sp1 = int(start_pos[0]), int(start_pos[1])
    xy = self._compute_pixel_resolution()
    margin = int(np.ceil(robot_radius / float(xy.min()))) + 1
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
      # Skip geoms whose center is well below the floor (likely visualisation
      # markers like *_target placed at z<0).
      if p[2] < -0.05:
        continue
      # World x → col (axis 1, width W); world y → row (axis 0, height H).
      c_col = (p[0] - wmin[0]) / (wmax[0] - wmin[0]) * W
      r_row = (p[1] - wmin[1]) / (wmax[1] - wmin[1]) * H
      rb_m = float(model.geom_rbound[gid])
      if rb_m <= 0:
        continue
      r_cells = max(1, int(np.ceil(rb_m / cell_min_m)))
      mask |= (yy - r_row)**2 + (xx - c_col)**2 <= r_cells**2
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

    exclude_patterns = ('robot0', 'mobilebase', 'gripper0', 'panda',
                        'eef_target', '_target', 'world',
                        'standing_table')
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
          return result if result is not None else _SAFE_FALLBACK_OBS
      except Exception:
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