from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing import Any, Optional, Callable

import sys
import os
import math
import numpy as np
import pickle
from dev_fn.util.console_io import RedirectStream, pformat, pprint

with RedirectStream(), RedirectStream(sys.stderr):
    from isaacgym import gymapi, gymutil, gymtorch
import torch

from . import embodiment_meta


class Embodiment:
    @staticmethod
    def __str__():
        pass

    @staticmethod
    def handle_asset_option(h_asset_options):
        pass

    @staticmethod
    def handle_asset_tendon(gym, asset, meta_info):
        pass

    @staticmethod
    def handle_asset_force_sensor(gym, asset, meta_info):
        pass

    @staticmethod
    def handle_actor_drive(h_dof_props, meta_info):
        pass


class Mano(Embodiment):
    @staticmethod
    def __str__():
        return "mano"

    @staticmethod
    def handle_actor_drive(h_dof_props, meta_info):
        stiffness = 1.00  # TODO: move to const
        damping = 0.10
        h_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        h_dof_props["stiffness"].fill(stiffness)
        h_dof_props["damping"].fill(damping)


class ShadowHand__NoForearm(Embodiment):
    @staticmethod
    def __str__():
        return "shadowhand_no_forearm"

    @staticmethod
    def handle_asset_option(h_asset_options):
        h_asset_options.angular_damping = 100
        h_asset_options.linear_damping = 100
        h_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)

    @staticmethod
    def handle_asset_tendon(gym, asset, meta_info):
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = gym.get_asset_tendon_properties(asset)
        for i in range(meta_info["num_tendon"]):
            for rt in relevant_tendons:
                if gym.get_asset_tendon_name(asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        gym.set_asset_tendon_properties(asset, tendon_props)

    @staticmethod
    def handle_asset_force_sensor(gym, asset, meta_info):
        fingertip_handles = [
            gym.find_asset_rigid_body_index(asset, name)
            for name in embodiment_meta.ShadowHand__NoForearm__Meta.fingertip_list
        ]
        sensor_pose = gymapi.Transform()
        for ft_handle in fingertip_handles:
            gym.create_asset_force_sensor(asset, ft_handle, sensor_pose)

    @staticmethod
    def handle_actor_drive(h_dof_props, meta_info):
        _h_drive = h_dof_props["driveMode"] == gymapi.DOF_MODE_POS
        # h_dof_props["stiffness"][_h_drive] *= 1
        # h_dof_props["effort"][_h_drive] *= 1


embodiment_mapping = {
    str(ShadowHand__NoForearm): ShadowHand__NoForearm,
}
