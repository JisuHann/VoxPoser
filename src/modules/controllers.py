import numpy as np
from utils.utils import get_logger

logger = get_logger(__name__)


class NavigationController:
    def __init__(self, env, config):
        self.config = config
        self.env = env

    def execute(self, waypoint):
        """
        Execute a navigation waypoint.
        :param waypoint: [target_xy, target_rotation, target_velocity]
        :return: info dict with mp_info
        """
        info = dict()
        result = self.env.apply_navigation_action(waypoint)
        info['mp_info'] = result
        return info
