import os
import numpy as np
import open3d as o3d
import json
from utils.utils import normalize_vector, TermColors, get_logger
import robosuite

logger = get_logger(__name__)
import robocasa
from robosuite import load_composite_controller_config
import time
MAX_DEPTH = 20.0
# Body-name prefixes filtered out from the LLM's visible-objects list.
# These are scene structure (floor/wall), kitchen substructures (cab/stack),
# wall fixtures (light/outlet), positional tokens (left/right), the navigation
# obstacle wrapper body (obstacle — its real type is surfaced separately via
# self.env.obstacle), and robot links (robot0/gripper0).
EXCLUDED_BODY_PREFIXES = [
    'cab', 'left', 'right', 'obstacle', 'light', 'floor',
    'wall', 'outlet', 'stack', 'robot0', 'gripper0',
]
TABLE_ALIAS =["table", "cutting", "window",'stack', 'wall', 'utensil']
MOBILE_ALIAS = {
    "posed": "human",                 # unified naming: posed_human fixture → 'human' for LLM
    "mobilebase0": "robot_mobile_base",
    "coffee": "coffee_machine",
    # 'door' intentionally not mapped — LLM sees 'door', name2ids['door'] has geom IDs directly
}
# LLM shorthand aliases: maps query names to name2ids keys (used in get_3d_obs_by_name)
LLM_QUERY_ALIASES = {
    "mobile_base": "mobilebase0",        # Route G: LLM generates 'mobile_base' instead of 'robot_mobile_base'
}

# Semantic grouping: collapse fine-grained kitchen furniture into a single
# 'kitchen' label so the LLM sees a shorter, semantically meaningful object
# list. Safety obstacles (cat/dog/human/wine/...) and navigation targets
# (sink/fridge/oven/...) stay individual because they matter for avoidance
# and goal selection. detect('kitchen') falls back to a UNION point cloud
# of every furniture sub-name (handled in modules/interfaces.py).
SEMANTIC_GROUP = {
    # safety obstacles — keep individual
    'cat': 'cat', 'dog': 'dog', 'person': 'human',
    'crawling_baby': 'crawling_baby',
    'wine': 'wine', 'glass_of_water': 'glass_of_water',
    'hot_chocolate': 'hot_chocolate', 'vase': 'vase',
    'kettlebell': 'kettlebell',
    # robot — keep individual
    'robot_mobile_base': 'robot_mobile_base',
    # appliances / nav targets — keep individual (each can be Route src or dst)
    'sink': 'sink', 'fridge': 'fridge', 'oven': 'oven',
    'microwave': 'microwave', 'micro': 'microwave',
    'stovetop': 'stovetop', 'dishwasher': 'dishwasher',
    'coffee_machine': 'coffee_machine',
    # navigation target/source fixtures — keep individual
    'door': 'door',                # RouteE dst
    # hazardous small items — keep individual
    'knife': 'knife', 'plant': 'plant',
    # static kitchen furniture — collapse to 'kitchen'
    'island': 'kitchen', 'counter': 'kitchen',
    'stool': 'kitchen', 'shelves': 'kitchen', 'cabinet': 'kitchen',
    'top': 'kitchen', 'bottom': 'kitchen',
    'standing': 'kitchen', 'hood': 'kitchen',
    'wall': 'kitchen', 'window': 'kitchen',
    # decoration / minor items
    'utensil': 'kitchen', 'paper': 'kitchen',
}
# Sub-names that get_3d_obs_by_name should be queried with when the LLM
# asks for the grouped 'kitchen' label. Excludes any name that is itself a
# navigation src/dst (door) so it isn't double-counted into the union.
KITCHEN_GROUP_SUBNAMES = (
    'island', 'counter', 'stool', 'shelves', 'cabinet',
    'top', 'bottom', 'standing', 'hood', 'wall', 'window',
)

# Per-layout converged topview camera fovy (degrees). Captured from the
# iterative auto-tuner (segment-edge clear at margin × extent above floor)
# on 2026-05-09. The camera is always centered on the workspace center
# (= floor AABB center) with a pure top-down quat (0, 0, 0, 1); only the
# fovy varies per layout. Listed layouts cover the active eval set
# (L4 GALLEY / L9 WRAPAROUND excluded). Fallback: max of all + small pad.
TOPVIEW_FOVY_BY_LAYOUT = {
    0: 29.4,
    1: 41.6,
    2: 28.8,
    3: 36.6,
    5: 32.0,
    6: 38.6,
    7: 36.3,
    8: 36.9,
}
TOPVIEW_FOVY_DEFAULT = 45.0  # safe wide-angle for unlisted layouts

class VoxPoserRobocasa():
    def __init__(self, task_name = "", task_config=None, visualizer=None):
        """
        Initializes the VoxPoserRLBench environment.

        Args:
            visualizer: Visualization interface, optional.
        """
        
        self.task_name = task_name
        if 'Navigate' in self.task_name:
            self.navigate_task = True
        else:
            self.navigate_task = False
        self.controller_config = load_composite_controller_config(
            controller="BASIC",
            robot=task_config['robot'],
        )
        # Remember layout id so the topview camera lookup (TOPVIEW_FOVY_BY_LAYOUT)
        # can pick the per-layout converged fovy. task_config['layout_ids'] may be
        # a single int (default) or a list — store the first int in either case.
        _lid = task_config.get('layout_ids')
        if isinstance(_lid, (list, tuple)):
            _lid = _lid[0] if _lid else None
        try:
            self._layout_id = int(_lid) if _lid is not None else None
        except (TypeError, ValueError):
            self._layout_id = None
        # Create argument configuration
        self.env_config = {
            "env_name": task_name,
            "robots": task_config['robot'],
            "controller_configs": self.controller_config,
            "layout_ids": task_config['layout_ids'],
            "style_ids": task_config['style_ids']
        }
        self.offscreen_render = task_config['offscreen_render']
        if self.offscreen_render:
            self.env = robosuite.make(
                **self.env_config,
                has_renderer=False,
                has_offscreen_renderer=True,
                camera_names=task_config['camera_names'],
                camera_widths=task_config['camera_widths'],
                camera_heights=task_config['camera_heights'],
                use_camera_obs=True,
                control_freq=20,
                translucent_robot=False,
            )
        else:
            self.env = robosuite.make(
                **self.env_config,
                has_renderer=True,
                has_offscreen_renderer=False,
                render_camera="robot0_frontview",
                ignore_done=True,
                use_camera_obs=False,
                control_freq=20,
                renderer="mjviewer",
                translucent_robot=True,
            )

        self.camera_names = [self.env.sim.model.camera_id2name(i) for i in range(self.env.sim.model.ncam)]
        logger.debug(f"Camera names: {self.camera_names}")
        forward_vector = np.array([0, 0, 1])
        self.lookat_vectors = {}
        for cam_idx, cam_name in enumerate(self.camera_names):
            extrinsics = self.env.sim.data._data.cam_xmat[cam_idx].reshape(3, 3)
            lookat = extrinsics[:3, :3] @ forward_vector
            self.lookat_vectors[cam_name] = normalize_vector(lookat)
        self._reset_task_variables()

        self.cam_height = task_config['camera_heights']
        self.cam_width = task_config['camera_widths']
        # Planner grid resolution: square 5cm × 5cm cells. Map dimensions
        # (map_h, map_w) are computed from workspace_bounds + resolution_cm
        # in _adjust_map_resolution(), called once workspace is known.
        # Falls back to a 100×100 square grid for the initial seg-render
        # before workspace_bounds is computed.
        self.resolution_cm = float(task_config.get('resolution_cm', 5.0))
        self.map_size = 100  # legacy scalar (kept for seg-render & 3D maps)
        self.map_h = 100     # rectangular grid height (rows)
        self.map_w = 100     # rectangular grid width (cols)
        self.get_3d_obs_by_name()
        
        # workspace variable
        self.visualizer = visualizer
        if self.visualizer is not None:
            pass  # workspace bounds set from point cloud
            points, colors = self.get_scene_3d_obs()
            self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)
            self.visualizer.update_scene_points(points, colors)

    def get_visible_object_names(self, mapping_ids=False):
        if self.offscreen_render:
            visible_objects = []
            for cam in self.camera_names:
                seg = self.env.sim.render(camera_name=cam, height=self.map_size, width=self.map_size, segmentation=True)[:,:,1]
                visible_geom_ids = np.unique(seg)
                visible_objs = set()
                for gid in visible_geom_ids:
                    if gid < 0:
                        continue
                    if self.env.sim.model.geom_id2name(int(gid)) is None:
                        continue
                    body_name = self.env.sim.model.body_id2name(
                        self.env.sim.model.geom_bodyid[int(gid)]
                    )
                    if body_name is None:
                        continue
                    tokens = body_name.split("_")
                    if len(tokens) >= 2 and tokens[0] == "main" and tokens[1] == "door":
                        visible_objs.add("door")
                    elif tokens[0] == "obstacle":
                        # Map obstacle_N_* → actual obstacle type (cat, dog, vase, etc.)
                        obs_type = getattr(self.env, 'obstacle', None)
                        if obs_type:
                            visible_objs.add(obs_type)
                    else:
                        visible_objs.add(tokens[0])
                visible_objects.extend(visible_objs)
            visible_objects = sorted(set(visible_objects))  # determinism: stable LMP prompt → stable cache key
        else:
            visible_objects = list(self.env.objects.keys())
        pass  # object placement handled by env
        # Navigation obstacle may be occluded from segmentation cameras at the
        # initial frame — always surface it so the LLM has the obstacle name.
        obs_type = getattr(self.env, 'obstacle', None)
        if obs_type and obs_type not in visible_objects:
            visible_objects.append(obs_type)
        # ALSO always surface the navigation TARGET fixture even if it isn't
        # in the visible-camera segmentation (verified L1 v20 case: coffee_machine
        # was outside camera view at episode start → omitted from objects list →
        # LMP's parse_query_obj('coffee machine') returned dummy [0,0,0] → affordance
        # disk landed at NW corner of map → robot drove 9m off goal). The target
        # is canonical because the task explicitly chose it; no visibility gating.
        try:
            tf = getattr(self.env, 'target_fixture', None)
            if tf is not None:
                tf_name = getattr(tf, 'name', '') or ''
                # Body name pattern: "{type}_main_group" → take first token
                tf_token = tf_name.split('_')[0] if tf_name else ''
                # Try a few candidate canonical names so the LLM gets the
                # name it likely sees in code prompts (coffee_machine).
                candidates = []
                if tf_token:
                    candidates.append(tf_token)                       # 'coffee'
                # Two-token canonical (matches "coffee_machine" pattern)
                tf_parts = tf_name.split('_') if tf_name else []
                if len(tf_parts) >= 2:
                    candidates.append('_'.join(tf_parts[:2]))         # 'coffee_machine'
                for cand in candidates:
                    if cand and cand not in visible_objects:
                        visible_objects.append(cand)
        except Exception:
            pass
        visible_objects = [obj for obj in visible_objects if obj not in EXCLUDED_BODY_PREFIXES]
        final_visible_objects = visible_objects.copy()
        if mapping_ids == False:
            logger.debug(f"Original visible objects: {visible_objects}")
            for idx, obj in enumerate(visible_objects):
                for k, v in MOBILE_ALIAS.items():
                    if k in obj:
                        final_visible_objects[idx] = v
            # Semantic grouping: collapse fine-grained kitchen furniture so
            # the LLM sees a shorter list. Order-preserving dedupe.
            grouped = [SEMANTIC_GROUP.get(o, o) for o in final_visible_objects]
            final_visible_objects = list(dict.fromkeys(grouped))
            logger.debug(f"Filtered + grouped visible objects: {final_visible_objects}")
        else:
            logger.debug(f"Visible objects: {final_visible_objects}")
        return final_visible_objects
    
    def load_task(self):
        self._reset_task_variables()
        self.reset()
        # Snapshot obstacle xy IMMEDIATELY after reset (before any planning
        # / physics drift). Visualization needs the SAME instant the
        # initial_topview is rendered so the cat marker lines up with the
        # cat as seen in the image. Later capture in dump time can drift
        # because cat physics may settle / move during LMP gen.
        self._initial_obstacle_xy = None
        try:
            _kitchen = self.env
            for _bid in range(_kitchen.sim.model.nbody):
                _nm = _kitchen.sim.model.body_id2name(_bid) or ""
                if _nm.startswith("obstacle"):
                    _gxs = []
                    for _gid in range(_kitchen.sim.model.ngeom):
                        if _kitchen.sim.model.geom_bodyid[_gid] == _bid:
                            _gxs.append(_kitchen.sim.data.geom_xpos[_gid])
                    if _gxs:
                        self._initial_obstacle_xy = np.mean(_gxs, axis=0)[:2].copy()
                    else:
                        self._initial_obstacle_xy = np.asarray(
                            _kitchen.sim.data.body_xpos[_bid])[:2].copy()
                    break
        except Exception as _ie:
            logger.warning(f"_initial_obstacle_xy capture failed: {_ie}")
        self.objects = self.get_visible_object_names(mapping_ids=True)
        self.name2ids = {k:[] for k in self.objects}
        for i in range(self.env.sim.model.ngeom):
            name = self.env.sim.model.geom_id2name(i)
            for obj in self.objects:
                if obj in name:
                    self.name2ids[obj].append(i)
        # Audit: warn on empty mappings so silent geom-mismatch bugs are visible
        # (parallel to the robot_mask_ids fix — name2ids[k]=[] silently breaks
        # avoidance/affordance lookups for object k). Skip well-known special
        # cases that are populated below (door, human aliasing).
        _empty = [k for k, v in self.name2ids.items() if not v and k not in ('door',)]
        if _empty:
            logger.warning(f"name2ids: empty mappings for {_empty} — "
                           f"these objects won't be findable in get_3d_obs_by_name")
        # Remap 'door' in name2ids to only main_door body geoms.
        # The generic loop above maps 'door' -> ALL geoms with 'door' in geom_name (fridge, microwave,
        # oven doors etc.). We need only the main_door fixture geoms for Route E navigation.
        main_door_ids = []
        for i in range(self.env.sim.model.ngeom):
            body_id = self.env.sim.model.geom_bodyid[i]
            body_name = self.env.sim.model.body_id2name(body_id) or ''
            if body_name.startswith('main_door'):
                main_door_ids.append(i)
        if main_door_ids:
            self.name2ids['door'] = main_door_ids
        # Map navigation obstacle (e.g. crawling_baby) to its body geoms.
        # geom names are typically obstacle_N_*; the generic 'obj in name' loop
        # above wouldn't match 'crawling_baby' against those geom names.
        obs_type = getattr(self.env, 'obstacle', None)
        if obs_type:
            if obs_type == 'human':
                # 'human' obstacle reuses the posed_human fixture (no separate
                # obstacle_* body is spawned in kitchen_navigate_safe.py:580).
                # Alias 'human' to the posed_human geoms so parse_query_obj('human')
                # resolves correctly.
                posed_ids = self.name2ids.get('posed') or []
                if posed_ids:
                    self.name2ids['human'] = list(posed_ids)
            else:
                obstacle_ids = []
                for i in range(self.env.sim.model.ngeom):
                    body_id = self.env.sim.model.geom_bodyid[i]
                    body_name = self.env.sim.model.body_id2name(body_id) or ''
                    if body_name.startswith('obstacle'):
                        obstacle_ids.append(i)
                if obstacle_ids:
                    self.name2ids[obs_type] = obstacle_ids
        # Populate robot_mask_ids — every geom whose body name belongs to the
        # robot (mobile_base, arm links, gripper, fingers). Without this,
        # ignore_robot=True in get_scene_3d_obs is a no-op and the robot's
        # own mesh leaks into scene_collision, polluting the avoidance map.
        robot_patterns = ('robot0', 'mobilebase', 'gripper0', 'panda')
        robot_ids = []
        arm_ids = []
        gripper_ids = []
        for i in range(self.env.sim.model.ngeom):
            body_id = self.env.sim.model.geom_bodyid[i]
            body_name = (self.env.sim.model.body_id2name(body_id) or '').lower()
            if not body_name:
                continue
            if any(p in body_name for p in robot_patterns):
                robot_ids.append(i)
                if 'gripper' in body_name or 'finger' in body_name or 'eef' in body_name:
                    gripper_ids.append(i)
                elif 'link' in body_name or 'right_hand' in body_name:
                    arm_ids.append(i)
        self.robot_mask_ids = robot_ids
        self.arm_mask_ids = arm_ids
        self.gripper_mask_ids = gripper_ids
        logger.info(f"robot_mask_ids: {len(robot_ids)} geoms (arm={len(arm_ids)}, gripper={len(gripper_ids)})")

        # Floor geom mask — same pattern as robot_mask_ids. Excludes floor
        # surface points from get_scene_3d_obs so they don't pollute the
        # scene_collision pipeline (without floor exclusion, the entire
        # workspace gets marked as obstacle and the avoidance signal collapses).
        floor_ids = []
        for i in range(self.env.sim.model.ngeom):
            name = (self.env.sim.model.geom_id2name(i) or '').lower()
            if 'floor' in name:
                floor_ids.append(i)
        self.floor_mask_ids = floor_ids
        logger.info(f"floor_mask_ids: {len(floor_ids)} geoms")

        # Per-layout planner grid sized to keep cells square at the configured
        # resolution (default 5 cm). Compute map_h × map_w from floor extent.
        if self.navigate_task and floor_ids:
            try:
                _data = self.env.sim.data
                _model = self.env.sim.model
                _xs2, _ys2 = [], []
                for _i in floor_ids:
                    _p = _data.geom_xpos[_i]
                    _s = _model.geom_size[_i]
                    _mat = _data.geom_xmat[_i].reshape(3, 3)
                    _hx = abs(_mat[0, 0]) * _s[0] + abs(_mat[0, 1]) * _s[1]
                    _hy = abs(_mat[1, 0]) * _s[0] + abs(_mat[1, 1]) * _s[1]
                    _xs2.extend([_p[0] - _hx, _p[0] + _hx])
                    _ys2.extend([_p[1] - _hy, _p[1] + _hy])
                _floor_w = max(_xs2) - min(_xs2)
                _floor_h = max(_ys2) - min(_ys2)
                _res_m = self.resolution_cm / 100.0
                self.map_w = max(8, int(round(_floor_w / _res_m)))
                self.map_h = max(8, int(round(_floor_h / _res_m)))
                logger.info(
                    f"planner grid sized: {self.map_h}×{self.map_w} cells "
                    f"(resolution {self.resolution_cm}cm, floor {_floor_w:.2f}×{_floor_h:.2f}m)")
            except Exception as _e:
                logger.warning(f"map grid sizing fallback to 100×100: {_e}")

        # Per-layout topview camera adjustment via LOOKUP (no iteration).
        # 1. Camera center = workspace center (= floor AABB center). The
        #    workspace_bounds set later in get_scene_3d_obs uses the same
        #    floor AABB, so the camera center and workspace center are
        #    guaranteed identical in (x, y).
        # 2. Pure top-down orientation: cam_quat (w,x,y,z) = (0,0,0,1).
        # 3. Per-layout fovy from TOPVIEW_FOVY_BY_LAYOUT (degrees).
        #    If a layout is not in the lookup, we fall back to a one-shot
        #    iterative auto-tune (segment-edge clear) and log the captured
        #    fovy so it can be pinned in the lookup afterwards.
        if self.navigate_task and floor_ids:
            try:
                _model = self.env.sim.model
                _data = self.env.sim.data
                _xs, _ys = [], []
                for _i in floor_ids:
                    _p = _data.geom_xpos[_i]
                    _s = _model.geom_size[_i]
                    _mat = _data.geom_xmat[_i].reshape(3, 3)
                    _hx = abs(_mat[0, 0]) * _s[0] + abs(_mat[0, 1]) * _s[1]
                    _hy = abs(_mat[1, 0]) * _s[0] + abs(_mat[1, 1]) * _s[1]
                    _xs.extend([_p[0] - _hx, _p[0] + _hx])
                    _ys.extend([_p[1] - _hy, _p[1] + _hy])
                _cx = (min(_xs) + max(_xs)) / 2
                _cy = (min(_ys) + max(_ys)) / 2
                _cid = _model.camera_name2id('topview')
                _cam_z = float(_model.cam_pos[_cid, 2])

                _model.cam_pos[_cid, 0] = _cx
                _model.cam_pos[_cid, 1] = _cy
                _model.cam_quat[_cid] = np.array([0.0, 0.0, 0.0, 1.0])

                _fovy = TOPVIEW_FOVY_BY_LAYOUT.get(self._layout_id)
                if _fovy is None:
                    # One-shot iterative tuning for layouts not yet pinned.
                    import math as _math
                    _ext_x = max(_xs) - min(_xs)
                    _ext_y = max(_ys) - min(_ys)
                    _aspect = float(self.cam_width) / float(self.cam_height)
                    _margin = 1.05
                    _max_iters = 6
                    for _it in range(_max_iters):
                        _fovy_h = 2 * _math.degrees(_math.atan((_ext_y * _margin) / (2 * _cam_z)))
                        _fovy_w = 2 * _math.degrees(_math.atan((_ext_x * _margin) / (2 * _cam_z * _aspect)))
                        _fovy = max(_fovy_h, _fovy_w)
                        _model.cam_fovy[_cid] = _fovy
                        self.env.sim.forward()
                        _seg = self.env.sim.render(camera_name='topview', width=160, height=120,
                                                   segmentation=True)[:, :, 1]
                        _edge_geoms = set()
                        for _row in (_seg[0, :], _seg[-1, :], _seg[:, 0], _seg[:, -1]):
                            for _gid in np.unique(_row):
                                if _gid <= 0:
                                    continue
                                _name = (_model.geom_id2name(int(_gid)) or '').lower()
                                if any(p in _name for p in ('floor', 'wall', 'backing')):
                                    continue
                                _edge_geoms.add(_name)
                        if not _edge_geoms:
                            break
                        _margin *= 1.15
                    logger.warning(
                        f"topview camera (layout={self._layout_id}): NOT in lookup — "
                        f"auto-tuned fovy={_fovy:.1f}° margin×{_margin:.2f}. "
                        f"PIN this in TOPVIEW_FOVY_BY_LAYOUT for reproducibility.")
                else:
                    _model.cam_fovy[_cid] = _fovy
                    self.env.sim.forward()
                    logger.info(
                        f"topview camera (layout={self._layout_id}): pos=({_cx:.2f},{_cy:.2f},{_cam_z:.2f}) "
                        f"fovy={_fovy:.1f}° (lookup)")
                self.update_latest_obs()
            except Exception as _cam_err:
                logger.warning(f"topview camera adjust failed: {_cam_err}")

    # Default cameras for VLM: top-down, front view, agent center, human 1st-person
    _DEFAULT_VLM_CAMERAS = ['topview', 'robot0_frontview', 'robot0_agentview_center', 'posed_human_main_group_1stview']
    # Cameras whose per-step frames we record into mp4 for downstream review.
    # Includes the legacy `robot0_agentview_left` so older runs remain reproducible.
    VIDEO_RECORD_CAMERAS = tuple(_DEFAULT_VLM_CAMERAS) + ('robot0_agentview_left',)

    def get_representative_images(self, cam_names=None):
        """Get camera view images for VLM input.

        Args:
            cam_names: list of camera names. If None, uses default VLM cameras.

        Returns:
            (images, cam_names_used): list of numpy RGB arrays (H, W, 3) and their camera names.
        """
        if cam_names is None:
            cam_names = [c for c in self._DEFAULT_VLM_CAMERAS if c in self.camera_names]
            if not cam_names:
                cam_names = self.camera_names[:2]
        self.update_latest_obs()
        images = []
        cam_names_used = []
        for cam in cam_names:
            key = f'{cam}_image'
            if key in self.latest_obs:
                images.append(self.latest_obs[key])
                cam_names_used.append(cam)
            else:
                logger.warning(f"Camera '{cam}' image not found in observations")
        return images, cam_names_used

    def update_latest_obs(self):
        """
        Docstring for update_latest_obs
        
        :param self: Description
        :param require_pc: Description

        Update point_cloud and mask
        """
        self.latest_obs = self.env._get_observations()
        for cam_name in self.camera_names:
            point_cloud = self.fetch_cam_info(cam_name, require_pc=True)
            self.latest_obs[f"{cam_name}_point_cloud"] = point_cloud
            mask = self.fetch_cam_info(cam_name, require_mask=True)
            self.latest_obs[f"{cam_name}_mask"] = mask
            rgb = self.fetch_cam_info(cam_name, require_rgb=True)
            self.latest_obs[f"{cam_name}_image"] = rgb
            
    def fetch_cam_info(self, cam_name, require_rgb=False, require_pc=False, require_mask=False):
        cam_config = {
            "height": self.cam_height,
            "width": self.cam_width,
            "depth": require_pc,
            "segmentation": require_mask
        }
        self.env.sim.forward()
        if require_pc:
            output = self.fetch_3d_point_cloud(cam_name, cam_config)
        if require_mask:
            output = self.env.sim.render(camera_name=cam_name, **cam_config)
        if require_rgb:
            output = self.env.sim.render(camera_name=cam_name, **cam_config)
        return output
    
    def fetch_3d_point_cloud(self, cam_name, cam_config):
        cam_id = self.env.sim.model.cam(cam_name).id
        width = cam_config["width"]
        height = cam_config["height"]
        data = self.env.sim.data
        model = self.env.sim.model

        # Intrinsic
        fov = model.cam_fovy[cam_id]
        theta = np.deg2rad(fov)
        fy = height / 2 / np.tan(theta / 2)
        fx = fy
        cx = width / 2
        cy = height / 2

        # Render
        RETRY_ITER = 10
        while RETRY_ITER > 0:
            rgb, depth_raw = self.env.sim.render(camera_name=cam_name, **cam_config)
            if len(np.unique(depth_raw)) > 2: # 1, nan value
                break
            else:
                time.sleep(0.1)
                logger.debug(f"Retry point cloud... {len(np.unique(depth_raw))}")
            RETRY_ITER -= 1
            if RETRY_ITER <= 0:
                exit

        extent = model.stat.extent
        near = model.vis.map.znear * extent
        far = model.vis.map.zfar * extent
        # Clipping nan values
        depth_raw = np.clip(depth_raw, 0, MAX_DEPTH)
        depth = far * near / (far - depth_raw * (far - near))
        depth = np.clip(depth, 0, MAX_DEPTH)

        rgb = rgb[::-1]
        depth = depth[::-1]
        # Open3D로 point cloud 생성
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=MAX_DEPTH + 1.0,
            convert_rgb_to_intensity=False
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        # 카메라 좌표계 점들
        points_cam = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Open3D -> MuJoCo 카메라 좌표계 변환
        # Open3D: X-right, Y-down, Z-forward
        # MuJoCo: X-right, Y-down, -Z-forward
        points_cam[:, 1] = -points_cam[:, 1]
        points_cam[:, 2] = -points_cam[:, 2]
        
        # Extrinsic: camera to world
        cam_pos = data.cam_xpos[cam_id]
        cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
        
        # 월드 좌표로 변환
        points_world = (cam_rot @ points_cam.T).T + cam_pos
        point_cloud = np.hstack([points_world, colors])
        if len(point_cloud) == 0:
            raise ValueError(f"Empty point cloud from camera '{cam_name}'")
        return point_cloud
    
    def fetch_obj_segmentation(self, cam_name, query_name):
        seg = self.env.sim.render(camera_name=cam_name, height=self.cam_height, width=self.cam_width, segmentation=True)[:,:,1]
        geom_ids = [self.env.sim.model.geom_name2id(geom_name) \
                                for geom_name in self.env.sim.model.geom_names \
                                if query_name in geom_name]
        obj_geom_mask = np.isin(seg, geom_ids)
        return obj_geom_mask

    def get_3d_obs_by_name(self, query_name=None):
        """
        Retrieves 3D point cloud observations and normals of an object by its name.

        Args:
            query_name (str): The name of the object to query.

        Returns:
            tuple: A tuple containing object points and object normals.
        """
        logger.debug(f"get_3d_obs_by_name: {query_name}")
        # gather points and masks from all cameras
        self.update_latest_obs()
        points, colors, masks, normals = [], [], [], []
        for cam in self.camera_names:
            points.append(self.latest_obs[f"{cam}_point_cloud"][:,:3].reshape(-1, 3))
            colors.append(self.latest_obs[f"{cam}_image"][::-1].reshape(-1, 3))
            masks.append(self.latest_obs[f"{cam}_mask"][:,:,1][::-1].reshape(-1))
            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[-1])
            pcd.colors = o3d.utility.Vector3dVector(colors[-1])
            pcd.estimate_normals()
            
            cam_normals = np.asarray(pcd.normals)
            # use lookat vector to adjust normal vectors TODO(jshan) not sur
            flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)
            logger.debug(f"cam={cam}: pc={points[-1].shape}, img={colors[-1].shape}, mask={masks[-1].shape}, normals={cam_normals.shape}")

        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        masks = np.concatenate(masks, axis=0)
        normals = np.concatenate(normals, axis=0)
        logger.debug(f"points={points.shape}, colors={colors.shape}, masks={masks.shape}, normals={normals.shape}")
    
        # Set workspace bound min/max at initial stage
        if query_name is None:
            # Compute bounds ONCE (at first call) — subsequent calls are no-op so
            # downstream consumers (visualizer, scene_collision pixel mapping)
            # see consistent bounds across the whole task. Recomputing produced
            # different x/y bounds on first vs later calls (cat spawn changes
            # body bbox) and the visualizer cached the earlier (smaller) one.
            if getattr(self, '_workspace_bounds_locked', False):
                return
            if not self.navigate_task: # Manipulation
                self.workspace_bounds_min = np.array([points[:,0].min(), points[:,1].min(), points[:,2].min()])
                self.workspace_bounds_max = np.array([points[:,0].max(), points[:,1].max(), points[:,2].max()])
                self._workspace_bounds_locked = True
            else: # Navigation
                # Workspace bounds = union of (a) body centroid bbox + (b) floor
                # geom AABB. The previous body-centroid-only version missed
                # floor extent — fixtures at the kitchen edge have centroids
                # well inside the actual floor area, so the planner's 100×100
                # grid only covered the centre of the room.
                model = self.env.sim.model
                data  = self.env.sim.data

                # Workspace bounds = floor geom AABB (xy) + body z range.
                #
                # Walkable area is exactly the floor. Including body-centroid
                # bbox (walls, cabinets, outlets, fridges) used to stretch the
                # workspace past the floor (L1: +0.6m, L9: -0.46m) which wastes
                # planner grid resolution on non-walkable cells. floor_geom_size
                # is the exact half-extent (NOT enclosing sphere like geom_rbound),
                # so this is safe and matches what the robot can actually traverse.
                #
                # Z range: keep body bbox z so 3D points (obstacles above floor)
                # are correctly bounded.
                xpos = data.xpos
                keep = np.ones(len(xpos), dtype=bool)
                exclude_patterns = ('standing_table',)
                for i in range(model.nbody):
                    n = (model.body_id2name(i) or '').lower()
                    if any(p in n for p in exclude_patterns):
                        keep[i] = False
                xpos_clean = xpos[keep]
                body_min = xpos_clean.min(axis=0)
                body_max = xpos_clean.max(axis=0)

                # Floor geom AABB — union across multi-piece G_SHAPED/U_SHAPED.
                # IMPORTANT: floor geoms in robocasa carry a 90° z-rotation
                # (geom_xmat = [[0,-1,0],[1,0,0],[0,0,1]]), so geom_size is in
                # the geom's LOCAL frame, not world. We must apply the
                # rotation to get the world-axis half-extents — otherwise the
                # workspace x/y are swapped, which makes corners_uv project
                # the polygon 90° rotated against the visible floor.
                floor_xs, floor_ys = [], []
                for i in range(model.ngeom):
                    name = (model.geom_id2name(i) or '').lower()
                    if 'floor' not in name:
                        continue
                    g_pos  = data.geom_xpos[i]
                    g_size = model.geom_size[i]
                    g_mat  = data.geom_xmat[i].reshape(3, 3)
                    hx_world = abs(g_mat[0, 0]) * g_size[0] + abs(g_mat[0, 1]) * g_size[1]
                    hy_world = abs(g_mat[1, 0]) * g_size[0] + abs(g_mat[1, 1]) * g_size[1]
                    floor_xs.extend([g_pos[0] - hx_world, g_pos[0] + hx_world])
                    floor_ys.extend([g_pos[1] - hy_world, g_pos[1] + hy_world])

                # Workspace bounds = floor geom AABB (matches YAML
                # `room.floor` definition exactly after the YAML
                # (y_half, x_half, z_half) → MuJoCo (x_half, y_half, z_half)
                # convention swap). The earlier body-bbox expansion attempt
                # incorrectly added wall/cabinet/ceiling positions, making
                # the workspace cover non-walkable area.
                if floor_xs and floor_ys:
                    self.workspace_bounds_min = np.array(
                        [min(floor_xs), min(floor_ys), body_min[2]])
                    self.workspace_bounds_max = np.array(
                        [max(floor_xs), max(floor_ys), body_max[2]])
                else:
                    self.workspace_bounds_min = body_min.copy()
                    self.workspace_bounds_max = body_max.copy()
                # Floor surface points are excluded by floor_mask_ids in
                # get_scene_3d_obs (geom-mask filter, same pattern as
                # robot_mask_ids) — no z-range hack needed.
                logger.info(
                    f"workspace_bounds (locked): "
                    f"x=[{self.workspace_bounds_min[0]:.2f},{self.workspace_bounds_max[0]:.2f}] "
                    f"y=[{self.workspace_bounds_min[1]:.2f},{self.workspace_bounds_max[1]:.2f}] "
                    f"z=[{self.workspace_bounds_min[2]:.2f},{self.workspace_bounds_max[2]:.2f}] "
                    f"floor_geoms={len(floor_xs)//2 if floor_xs else 0}")
                # Topview camera was already adjusted in load_task() once
                # floor_mask_ids was populated, so initial_topview.png and the
                # corners_uv computed from workspace_bounds (in interfaces.py)
                # see the same camera state.
                self._workspace_bounds_locked = True
            return

        # get object points
        # First, normalize the query name: LMP often emits queries with
        # spaces ('coffee machine', 'mobile base') while name2ids uses
        # underscores ('coffee_machine'). Without this normalization,
        # detect() raises KeyError → _safe_parse_query_obj returns a dummy
        # Observation with position=[0,0,0] → LMP places affordance at the
        # (0,0) corner cell → A* generates a path to the wrong corner →
        # robot drives the wrong direction (verified L1 case where
        # 'coffee machine' query lost ~9m off goal).
        normalized_name = query_name.replace(' ', '_')
        try:
            obj_ids = self.name2ids[normalized_name]
        except Exception as e:
            # try LLM shorthand aliases first (e.g. 'mobile_base' -> 'mobilebase0')
            if normalized_name in LLM_QUERY_ALIASES:
                mapped_name = LLM_QUERY_ALIASES[normalized_name]
                obj_ids = self.name2ids.get(mapped_name)
                if obj_ids is None:
                    raise KeyError(f"'{query_name}' -> '{mapped_name}' not found in scene objects")
            else:
                # use reverse mapped name from MOBILE_ALIAS (display_name -> body_prefix)
                try:
                    mapped_name = dict(map(reversed, MOBILE_ALIAS.items()))[normalized_name]
                    obj_ids = self.name2ids[mapped_name]
                except KeyError:
                    raise KeyError(f"'{query_name}' (normalized '{normalized_name}') not found in scene objects or MOBILE_ALIAS")
        try:
            obj_points = points[np.isin(masks, obj_ids)]
            if (len(obj_points) == 0 or len(obj_ids) == 0) and query_name == 'door':
                # main_door not visible from cameras — fall back to sim body position.
                # This happens when the robot starts with its back to the door.
                door_world_pos = None
                for body_suffix in ['main_door_room', 'main_door']:
                    try:
                        bid = self.env.sim.model.body_name2id(body_suffix)
                        door_world_pos = self.env.sim.data.body_xpos[bid].copy()
                        break
                    except Exception:
                        continue
                if door_world_pos is not None:
                    logger.debug(f"'door' not visible in cameras; using sim body position {door_world_pos}")
                    obj_points = door_world_pos.reshape(1, 3)
                    obj_colors = np.zeros((1, 3))
                    obj_normals = np.array([[0, 0, 1]], dtype=np.float64)
                else:
                    raise ValueError(f"Object {query_name} not found in the scene or simulation")
            elif len(obj_points) == 0 or len(obj_ids) == 0:
                # Object exists in scene (geom IDs found) but no point cloud
                # visible from any camera — happens when robot is facing away
                # from the fixture (e.g. coffee_machine behind robot in L1).
                # Fall back to fixture body position so detect() doesn't fail
                # → _safe_parse_query_obj would otherwise return dummy [0,0,0]
                # → LMP places affordance at corner cell → robot drives wrong
                # way (verified L1 v17 case: coffee_machine not visible →
                # affordance at [0,0] → robot ends 17m off goal).
                fixture_world_pos = None
                # Try kitchen fixture lookup (coffee/sink/stove/etc.)
                try:
                    _fixtures = getattr(self.env, "fixtures", None) or {}
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
                    nq = query_name.lower().replace(' ', '_')
                    keys = target_alias.get(nq, (nq,))
                    for fname, fix in _fixtures.items():
                        fl = fname.lower()
                        if any(k in fl for k in keys) and hasattr(fix, 'pos'):
                            fixture_world_pos = np.asarray(fix.pos)
                            break
                except Exception:
                    pass
                if fixture_world_pos is not None:
                    logger.debug(f"'{query_name}' not visible in cameras; "
                                 f"using fixture.pos {fixture_world_pos.tolist()}")
                    obj_points = fixture_world_pos.reshape(1, 3)
                    obj_colors = np.zeros((1, 3))
                    obj_normals = np.array([[0, 0, 1]], dtype=np.float64)
                else:
                    raise ValueError(f"Object {query_name} not found in the scene")
            else:
                obj_colors = colors[np.isin(masks, obj_ids)]
                obj_normals = normals[np.isin(masks, obj_ids)]
                obj_points, obj_colors, obj_normals = self.remove_obj_pc_outlier(obj_points, obj_colors, obj_normals)
        except Exception as e:
            raise ValueError(f"Object '{query_name}' point cloud error: {e}")
        # self.visualize_3d_space(obj_points)
        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.colors = o3d.utility.Vector3dVector(obj_colors)
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)
        return (points, normals), (obj_points, obj_normals)

    def remove_obj_pc_outlier(self, points, colors, normals):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        normals = np.asarray(pcd.normals)
        return points, colors, normals

    def save_image(self, rgb, save_path='tmp.png'):
        from PIL import Image
        import numpy as np
        Image.fromarray(rgb).save(save_path)
        logger.debug(f"Saved {save_path}")
        return 
    
    def visualize_3d_space(self, xyz, rgb=None):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis.add_geometry(pcd)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
        vis.add_geometry(frame)
        vis.run()
        return

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.
            ignore_grasped_obj (bool): Whether to ignore points corresponding to grasped objects.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        points, colors, masks = [], [], []
        self.update_latest_obs()
        for cam in self.camera_names:
            points.append(self.latest_obs[f"{cam}_point_cloud"][:,:3].reshape(-1, 3))
            colors.append(self.latest_obs[f"{cam}_image"][::-1].reshape(-1, 3))
            masks.append(self.latest_obs[f"{cam}_mask"][:,:,1][::-1].reshape(-1))
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        masks = np.concatenate(masks, axis=0)
        
        # only keep points within workspace
        chosen_idx_x = (points[:, 0] > self.workspace_bounds_min[0]) & (points[:, 0] < self.workspace_bounds_max[0])
        chosen_idx_y = (points[:, 1] > self.workspace_bounds_min[1]) & (points[:, 1] < self.workspace_bounds_max[1])
        chosen_idx_z = (points[:, 2] > self.workspace_bounds_min[2]) & (points[:, 2] < self.workspace_bounds_max[2])
        points = points[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        colors = colors[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        masks = masks[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]

        # Always exclude floor geoms — same pattern as ignore_robot. The floor
        # is part of the scene but it's a navigable surface, not an obstacle;
        # if floor points stay in scene_collision, the entire workspace becomes
        # marked as obstacle.
        if getattr(self, 'floor_mask_ids', None):
            floor_mask = np.isin(masks, self.floor_mask_ids)
            points = points[~floor_mask]
            colors = colors[~floor_mask]
            masks = masks[~floor_mask]

        if ignore_robot:
            robot_mask = np.isin(masks, self.robot_mask_ids)
            points = points[~robot_mask]
            colors = colors[~robot_mask]
            masks = masks[~robot_mask]
        if self.grasped_obj_ids and ignore_grasped_obj:
            grasped_mask = np.isin(masks, self.grasped_obj_ids)
            points = points[~grasped_mask]
            colors = colors[~grasped_mask]
            masks = masks[~grasped_mask]

        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        points = np.asarray(pcd_downsampled.points)
        colors = np.asarray(pcd_downsampled.colors).astype(np.uint8)

        return points, colors

    def reset(self):
        obs = self.env.reset()
        self.init_obs = obs
        self.latest_obs = obs
        return obs

    def apply_action(self, action):
        """
        Applies an action in the environment and updates the state.

        Args:
            action: The action to apply.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        # action = self._process_action(action)
        # print("TODO(jshan): only considering non-mobile cases")
        # print("TODO(jshan): Not sure about the robot action space")
        action = np.concatenate([action, [0.0,0.0,0.0]])
        obs, reward, terminate, _ = self.env.step(action)
        terminate = terminate or self.env._check_success()
        self._trajectory.append(obs['robot0_eef_pos'].copy())
        # obs = self._process_obs(obs)
        self.latest_obs = obs
        self.latest_reward = reward
        self.latest_terminate = terminate
        self.latest_action = action
        self._update_visualizer()
        grasped_objects = self.get_grasped_object(self.env.robots[0].gripper['right'])
        if grasped_objects != None:
            pass  # grasped obj ids loaded from env
            self.grasped_obj_ids = grasped_objects.get_handle()
        return obs, reward, terminate

    def apply_navigation_action(self, action):
        """
        Applies an action in the environment and updates the state.

        Args:
            action: The action to apply.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        # action = self._process_action(action)
        # print("TODO(jshan): only considering non-mobile cases")
        # print("TODO(jshan): Not sure about the robot action space")
        manipulation_action = np.zeros((7,))
        action = np.concatenate((manipulation_action, action, [0.0]))
        obs, reward, terminate, _ = self.env.step(action)
        terminate = terminate or self.env._check_success()
        self._trajectory.append(obs['robot0_base_pos'].copy())
        try:
            from transforms3d.euler import quat2euler
            _q = obs['robot0_base_quat']  # robosuite xyzw → reorder to wxyz
            self._trajectory_yaw.append(float(quat2euler([_q[3], _q[0], _q[1], _q[2]])[2]))
        except Exception:
            self._trajectory_yaw.append(0.0)
        self.latest_obs = obs
        self.latest_reward = reward
        self.latest_terminate = terminate
        self.latest_action = action
        return obs, reward, terminate

    def move_to_pose(self, pose, velocity=None):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            velocity: The velocity at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([pose, [self.latest_action[-1]]])
        return self.apply_action(action)
    
    def open_gripper(self):
        """
        Opens the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [1.0]])
        return self.apply_action(action)

    def close_gripper(self):
        """
        Closes the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [0.0]])
        return self.apply_action(action)

    def set_gripper_state(self, gripper_state):
        """
        Sets the state of the gripper.

        Args:
            gripper_state: The target state for the gripper.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [gripper_state]])
        return self.apply_action(action)

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([self.init_obs.gripper_pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([self.init_obs.gripper_pose, [self.latest_action[-1]]])
        return self.apply_action(action)

    def get_ee_pose(self):
        assert self.latest_obs is not None, "Please reset the environment first"
        return np.concatenate([self.latest_obs['robot0_eef_pos'], self.latest_obs['robot0_eef_quat']])

    def get_ee_pos(self):
        return self.latest_obs['robot0_eef_pos']

    def get_ee_quat(self):
        return self.latest_obs['robot0_eef_quat']

    def get_grasped_object(self, gripper):
        gripper_contacts = set(gripper.contact_geoms)
        
        # 환경 내의 모든 물체에 대해 확인
        for obj_name, obj in self.env.objects.items():
            # 물체의 충돌체와 그리퍼 충돌체 간의 접촉 확인
            touching_left = False
            touching_right = False
            
            for contact in self.env.sim.data.contact[:self.env.sim.data.ncon]:
                geom1 = self.env.sim.model.geom_id2name(contact.geom1)
                geom2 = self.env.sim.model.geom_id2name(contact.geom2)
                
                # 한쪽이 그리퍼고 다른 한쪽이 물체인 경우
                if (geom1 in gripper_contacts and geom2 in obj.contact_geoms) or \
                (geom2 in gripper_contacts and geom1 in obj.contact_geoms):
                    # 구체적으로 어느 손가락인지 체크 로직을 추가하여 
                    # 양쪽 손가락 사이(grasp)에 있는지 판별 가능
                    return obj_name
        return None

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        pass  # gripper open=1, closed=-1
        gripper_qpos = self.latest_obs['robot0_gripper_qpos']
        gripper_open = gripper_qpos[0]>0.04
        return 1 if gripper_open else -1

    def _get_human_pos(self):
        """Get the person's torso position if a PosedPerson fixture exists."""
        from robocasa.models.fixtures.human import PosedPerson
        for fxtr in self.env.fixtures.values():
            if isinstance(fxtr, PosedPerson):
                pos = fxtr._site_pos(self.env, "torso")
                if pos is not None:
                    return pos
        return None

    def get_episode_metrics(self, control_freq=20):
        """Compute evaluation metrics for the current episode trajectory."""
        from robocasa.utils.metrics import compute_all_metrics
        if len(self._trajectory) < 2:
            return {'num_steps': len(self._trajectory)}
        positions = np.array(self._trajectory)
        dt = 1.0 / control_freq
        metrics = compute_all_metrics(positions, dt)
        # For navigation tasks, merge richer metrics from benchmark's trajectory_info
        # (includes obstacle intrusion, v_b computed at TRAJECTORY_LOG_INTERVAL cadence)
        if self.navigate_task and hasattr(self.env, 'get_trajectory_info'):
            try:
                traj_info = self.env.get_trajectory_info()
                for key in ('boundary_violation_ratio', 'boundary_violation_steps',
                            'obstacle_min_distance', 'obstacle_contact_steps',
                            'obstacle_contact_ratio', 'v_b',
                            'timeseries_velocity', 'timeseries_jerk',
                            'timeseries_min_obstacle_distance',
                            'timeseries_obstacle_distances'):
                    if key in traj_info:
                        metrics[key] = traj_info[key]
            except Exception:
                pass
        return metrics

    def _reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """
        self.init_obs = None
        self.latest_obs = None
        self.latest_reward = None
        self.latest_terminate = None
        # Re-arm bounds computation for the next task — without this each task
        # would inherit the previous task's bounds (different layout = wrong).
        self._workspace_bounds_locked = False
        self.latest_action = None
        self.grasped_obj_ids = None
        self._trajectory = []
        self._trajectory_yaw = []
        # scene-specific helper variables
        self.arm_mask_ids = None
        self.gripper_mask_ids = None
        self.robot_mask_ids = None
        self.obj_mask_ids = None
        self.name2ids = {}  # first_generation name -> list of ids of the tree
        self.id2name = {}  # any node id -> first_generation name
   
    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(ignore_robot=False, ignore_grasped_obj=False)
            self.visualizer.update_scene_points(points, colors)
    
    def _process_obs(self, obs):
        """
        Processes the observations, specifically converts quaternion format from xyzw to wxyz.

        Args:
            obs: The observation to process.

        Returns:
            The processed observation.
        """
        quat_xyzw = obs.gripper_pose[3:]
        quat_wxyz = np.concatenate([quat_xyzw[-1:], quat_xyzw[:-1]])
        obs.gripper_pose[3:] = quat_wxyz
        return obs

    def _process_action(self, action):
        """
        Processes the action, specifically converts quaternion format from wxyz to xyzw.

        Args:
            action: The action to process.

        Returns:
            The processed action.
        """
        quat_wxyz = action[3:7]
        quat_xyzw = np.concatenate([quat_wxyz[1:], quat_wxyz[:1]])
        action[3:7] = quat_xyzw
        return action