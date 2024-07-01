from __future__ import annotations
import sys
import os
import numpy as np
from dev_fn.util.console_io import RedirectStream

with RedirectStream(), RedirectStream(sys.stderr):
    from isaacgym import gymapi, gymutil, gymtorch

import torch
import logging
import typing

if typing.TYPE_CHECKING:
    from .general import EnvGeneral


from .common import EnvCallbackContext
from .tool import forward_kinematic_chain, compute_contact_wordaround
from ..scene import GeomCata
import pytorch_kinematics

from dev_fn.util.console_io import pprint
from dev_fn.transform.rotation import euler_angle_to_rotmat, rotmat_to_euler_angle, quat_to_rotmat, rotmat_to_quat
from dev_fn.transform.transform import inv_transf, transf_point_array, assemble_T
from dev_fn.transform.transform import inv_rotmat, rotate_point_array

from ..util.torch_jit_util import scale, tensor_clamp

_logger = logging.getLogger(__name__)


class OakInk2DevEnvCallbackContext(EnvCallbackContext):
    def __init__(self, num_env: int, device: torch.device, dtype: torch.dtype):
        self.num_env = num_env
        self.device = device
        self.dtype = dtype

        self.lh_enabled, self.rh_enabled = False, True

        self.act_moving_average = 1.0

        self.transition_scale = 0.5
        self.orientation_scale = 0.1

        self.reset()

    @property
    def obs_dim(self):
        return 300

    @property
    def act_dim(self):
        return 24

    @property
    def timeout(self):
        return 10  # seconds

    def reset(self):
        self.rh_dof_index_arr = None
        self.rh_root_rb_index_arr = None

        self.dt = None
        self.num_dof = None
        self.target_prev = None
        self.target_curr = None
        self.apply_force = None
        self.apply_torque = None

        self.actuated_dof_indices = None
        self.shadow_hand_dof_lower_limits = None
        self.shadow_hand_dof_upper_limits = None

    def attach_env(self, env: EnvGeneral):
        super().attach_env(env)
        assert self.num_env == self.env.num_envs

        # region: index
        rh_dof_index_arr = [] if self.rh_enabled else None
        rh_root_rb_index_arr = [] if self.rh_enabled else None
        for env_id in range(self.num_env):
            # dof
            _curr_dof_index_begin = self.env.dof_index_begin_list[env_id]
            _curr_dof_index_store = self.env.env_dof_index_store[env_id]
            if self.rh_enabled:
                _curr_local_rh_dof_index = _curr_dof_index_store["rh"]
                _curr_rh_dof_index = _curr_dof_index_begin + _curr_local_rh_dof_index
                rh_dof_index_arr.append(_curr_rh_dof_index)

            # rb
            _curr_rb_index_begin = self.env.rb_index_begin_list[env_id]
            _curr_rb_index_map_store = self.env.env_rb_index_map_store[env_id]
            if self.rh_enabled:
                _curr_rh_root_rb_index = _curr_rb_index_begin + _curr_rb_index_map_store["rh"]["robot0:palm"]
                rh_root_rb_index_arr.append(_curr_rh_root_rb_index)
        if self.rh_enabled:
            self.rh_dof_index_arr = torch.as_tensor(
                np.stack(rh_dof_index_arr, axis=0), device=self.device, dtype=torch.long
            )
            self.rh_root_rb_index_arr = torch.as_tensor(rh_root_rb_index_arr, device=self.device, dtype=torch.long)
        else:
            self.rh_dof_index_arr = None
            self.rh_root_rb_index_arr = None
        # endregion: index

        self.dt = self.env.dt
        self.num_dof = self.rh_dof_index_arr.shape[1]
        self.target_prev = torch.zeros((self.num_env, self.num_dof), dtype=self.dtype, device=self.device)
        self.target_curr = torch.zeros((self.num_env, self.num_dof), dtype=self.dtype, device=self.device)
        self.target_dof_pos = torch.zeros_like(self.env.dof_pos)
        self.apply_force = torch.zeros((self.env.rigid_body_state.shape[0], 3), dtype=self.dtype, device=self.device)
        self.apply_torque = torch.zeros((self.env.rigid_body_state.shape[0], 3), dtype=self.dtype, device=self.device)

        self.actuated_dof_indices = torch.as_tensor(
            self.env.meta_info["rh"]["actuated_dof_index"], dtype=torch.long, device=self.device
        )
        self.shadow_hand_dof_lower_limits = torch.as_tensor(
            self.env.limit_info["rh"]["lower"], dtype=self.dtype, device=self.device
        )
        self.shadow_hand_dof_upper_limits = torch.as_tensor(
            self.env.limit_info["rh"]["upper"], dtype=self.dtype, device=self.device
        )

    def detach_env(self):
        super().detach_env()
        self.reset()

    # action
    def apply_action(self, action: torch.Tensor, env_id_to_reset: torch.Tensor):
        # fill target_curr and target_prev for env_id_to_reset
        if len(env_id_to_reset) > 0:
            self.target_prev[env_id_to_reset, :] = self.env.dof_pos[self.rh_dof_index_arr[env_id_to_reset]].squeeze(-1)
            self.target_curr[env_id_to_reset, :] = self.env.dof_pos[self.rh_dof_index_arr[env_id_to_reset]].squeeze(-1)

        self.target_curr[:, self.actuated_dof_indices] = scale(
            action[:, 6:],
            self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
            self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
        )
        self.target_curr[:, self.actuated_dof_indices] = (
            self.act_moving_average * self.target_curr[:, self.actuated_dof_indices]
            + (1.0 - self.act_moving_average) * self.target_prev[:, self.actuated_dof_indices]
        )
        self.target_curr[:, self.actuated_dof_indices] = tensor_clamp(
            self.target_curr[:, self.actuated_dof_indices],
            self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
            self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
        )

        self.apply_force[self.rh_root_rb_index_arr, :] = action[:, 0:3] * self.dt * self.transition_scale * 100000 # ORI: 100000
        self.apply_torque[self.rh_root_rb_index_arr, :] = (
            action[:, 3:6] * self.dt * self.orientation_scale * 10000
        )  # ORI: 1000
        self.env.gym.apply_rigid_body_force_tensors(
            self.env.sim,
            gymtorch.unwrap_tensor(self.apply_force),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )
        
        self.target_prev[:, self.actuated_dof_indices] = self.target_curr[:, self.actuated_dof_indices]
        self.target_dof_pos[self.rh_dof_index_arr] = self.target_prev.unsqueeze(-1)
        _index_arr = self.env.rh_index_arr.to(torch.int32)
        self.env.gym.set_dof_position_target_tensor_indexed(
            self.env.sim,
            gymtorch.unwrap_tensor(self.target_dof_pos),
            gymtorch.unwrap_tensor(_index_arr),
            len(_index_arr),
        )

    def compute_observation(self):
        obs = torch.zeros((self.num_env, self.obs_dim), dtype=self.dtype, device=self.device)
        self.env.obs_buf[:, :] = obs

    def compute_reward(self, action: torch.Tensor):
        # handle reset
        resets = self.env.reset_buf
        resets = torch.where(self.env.progress_buf >= self.env.max_episode_length, torch.ones_like(resets), resets)
        self.env.reset_buf[:] = resets
        
        # handle rew
        self.env.rew_buf[:] = torch.zeros((self.num_env, ), dtype=self.dtype, device=self.device)

    def viz_debug(self):
        self.env.gym.clear_lines(self.env.viewer)
