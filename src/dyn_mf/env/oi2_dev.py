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
from ..scene import GeomCata
from .embodiment import ShadowHand__NoForearm
from .embodiment_meta import ShadowHand__NoForearm__Meta
import pytorch_kinematics

from dev_fn.util.console_io import pprint
from dev_fn.transform.rotation_np import rotmat_to_quat_np, quat_to_rotmat_np, quat_apply_np
from dev_fn.transform.transform_np import assemble_T_np, inv_transf_np
from dev_fn.transform.rotation_jit import (
    euler_angle_to_rotmat,
    rotmat_to_euler_angle,
    quat_to_rotmat,
    rotmat_to_quat,
    quat_apply,
    quat_invert,
    quat_multiply,
    quat_to_euler_angle,
)
from dev_fn.transform.transform_jit import inv_transf, transf_point_array, assemble_T
from dev_fn.transform.transform_jit import inv_rotmat, rotate_point_array
from dev_fn.geo.ops.check_sign import check_sign

from ..util.torch_jit_util import scale, unscale, tensor_clamp
from ..util import isaacgym_util
from ..util.isaacgym_util import debug_viz_frame

_logger = logging.getLogger(__name__)


class OakInk2DevEnvCallbackContext(EnvCallbackContext):

    def __init__(
        self,
        num_env: int,
        device: torch.device,
        dtype: torch.dtype,
        enable_evaluation: bool = False,
        evaluation_success_threshold: float = 0.50,
    ):
        self.num_envs = num_env
        self.device = device
        self.dtype = dtype

        self.lh_enabled, self.rh_enabled = False, True

        self.act_moving_average = 1.0

        self.translation_scale = 0.5
        self.orientation_scale = 0.1

        self.enable_evaluation = enable_evaluation
        self.evaluation_success_threshold = evaluation_success_threshold

        self.fingertips = ShadowHand__NoForearm__Meta.fingertip_list
        self.num_fingertips = len(self.fingertips)
        self.hand_center = ["robot0:palm"]

        self.clear_buf()

    @property
    def obs_dim(self):
        return 300

    @property
    def act_dim(self):
        return 24

    @property
    def timeout(self):
        return 10  # seconds

    def clear_buf(self):
        self.rh_dof_index_arr = None
        self.rh_palm_rb_index_arr = None
        self.rh_ff_rb_index_arr = None
        self.rh_mf_rb_index_arr = None
        self.rh_rf_rb_index_arr = None
        self.rh_lf_rb_index_arr = None
        self.rh_th_rb_index_arr = None
        self.rh_fingertip_rb_index_arr = None

        self.dt = None
        self.num_dof = None
        self.target_prev = None
        self.target_curr = None
        self.apply_force = None
        self.apply_torque = None

        self.actuated_dof_indices = None
        self.shadow_hand_dof_lower_limits = None
        self.shadow_hand_dof_upper_limits = None

        if self.enable_evaluation:
            self.fluid_particle_index_arr_list = None
            self.fluid_particle_tsl_list = None
            self.obj_dst_id_list = None
            self.obj_dst_index_arr = None
            self.obj_dst_tsl = None
            self.obj_dst_quat = None
            self.obj_dst_interior_store = None

    def _collect_index_rh(self):
        rh_dof_index_arr = [] if self.rh_enabled else None
        rh_palm_rb_index_arr = [] if self.rh_enabled else None
        rh_ff_rb_index_arr = [] if self.rh_enabled else None
        rh_mf_rb_index_arr = [] if self.rh_enabled else None
        rh_rf_rb_index_arr = [] if self.rh_enabled else None
        rh_lf_rb_index_arr = [] if self.rh_enabled else None
        rh_th_rb_index_arr = [] if self.rh_enabled else None
        rh_fingertip_rb_index_arr = [] if self.rh_enabled else None
        rh_fs_index_arr = [] if self.rh_enabled else None
        for env_id in range(self.num_envs):
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
                _curr_rh_palm_rb_index = _curr_rb_index_begin + _curr_rb_index_map_store["rh"]["robot0:palm"]
                rh_palm_rb_index_arr.append(_curr_rh_palm_rb_index)
                _curr_rh_ff_rb_index = _curr_rb_index_begin + _curr_rb_index_map_store["rh"]["robot0:ffdistal"]
                rh_ff_rb_index_arr.append(_curr_rh_ff_rb_index)
                _curr_rh_mf_rb_index = _curr_rb_index_begin + _curr_rb_index_map_store["rh"]["robot0:mfdistal"]
                rh_mf_rb_index_arr.append(_curr_rh_mf_rb_index)
                _curr_rh_rf_rb_index = _curr_rb_index_begin + _curr_rb_index_map_store["rh"]["robot0:rfdistal"]
                rh_rf_rb_index_arr.append(_curr_rh_rf_rb_index)
                _curr_rh_lf_rb_index = _curr_rb_index_begin + _curr_rb_index_map_store["rh"]["robot0:lfdistal"]
                rh_lf_rb_index_arr.append(_curr_rh_lf_rb_index)
                _curr_rh_th_rb_index = _curr_rb_index_begin + _curr_rb_index_map_store["rh"]["robot0:thdistal"]
                rh_th_rb_index_arr.append(_curr_rh_th_rb_index)
                _curr_fingertip_rb_index_list = []
                for fingertip_name in self.fingertips:
                    _curr_fingertip_rb_index = _curr_rb_index_begin + _curr_rb_index_map_store["rh"][fingertip_name]
                    _curr_fingertip_rb_index_list.append(_curr_fingertip_rb_index)
                rh_fingertip_rb_index_arr.append(_curr_fingertip_rb_index_list)

            # fs
            _curr_fs_index_begin = self.env.force_sensor_index_begin_list[env_id]
            _curr_fs_index_store = self.env.env_force_sensor_index_store[env_id]
            if self.rh_enabled:
                _curr_local_rh_fs_index = _curr_fs_index_store["rh"]
                _curr_rh_fs_index = _curr_fs_index_begin + _curr_local_rh_fs_index
                rh_fs_index_arr.append(_curr_rh_fs_index)

        if self.rh_enabled:
            self.rh_dof_index_arr = torch.as_tensor(
                np.stack(rh_dof_index_arr, axis=0), device=self.device, dtype=torch.long
            )
            self.rh_palm_rb_index_arr = torch.as_tensor(rh_palm_rb_index_arr, device=self.device, dtype=torch.long)
            self.rh_ff_rb_index_arr = torch.as_tensor(rh_ff_rb_index_arr, device=self.device, dtype=torch.long)
            self.rh_mf_rb_index_arr = torch.as_tensor(rh_mf_rb_index_arr, device=self.device, dtype=torch.long)
            self.rh_rf_rb_index_arr = torch.as_tensor(rh_rf_rb_index_arr, device=self.device, dtype=torch.long)
            self.rh_lf_rb_index_arr = torch.as_tensor(rh_lf_rb_index_arr, device=self.device, dtype=torch.long)
            self.rh_th_rb_index_arr = torch.as_tensor(rh_th_rb_index_arr, device=self.device, dtype=torch.long)
            self.rh_fingertip_rb_index_arr = torch.as_tensor(
                rh_fingertip_rb_index_arr, device=self.device, dtype=torch.long
            )
            self.rh_fs_index_arr = torch.as_tensor(
                np.stack(rh_fs_index_arr, axis=0), device=self.device, dtype=torch.long
            )
        else:
            self.rh_dof_index_arr = None
            self.rh_palm_rb_index_arr = None
            self.rh_ff_rb_index_arr = None
            self.rh_mf_rb_index_arr = None
            self.rh_rf_rb_index_arr = None
            self.rh_lf_rb_index_arr = None
            self.rh_th_rb_index_arr = None
            self.rh_fingertip_rb_index_arr = None
            self.rh_fs_index_arr = None

    def _collect_index_fluid(self):
        # use list as each environment may have differing number of fluid particle
        self.fluid_particle_index_arr_list = []
        from dev_fn.util.console_io import pprint

        for env_id in range(self.num_envs):
            fluid_index_arr = []
            num_fluid_particle = self.env.init_asset["evaluation_info"][env_id]["num_fluid_particle"]
            _curr_obj_aux_index_store = self.env.obj_aux_index_store[env_id]
            for particle_id in range(num_fluid_particle):
                fluid_index_arr.append(_curr_obj_aux_index_store[f"fluid_{particle_id:0>3}"])
            fluid_index_arr = torch.as_tensor(fluid_index_arr, dtype=torch.long, device=self.device)
            self.fluid_particle_index_arr_list.append(fluid_index_arr)

    def _collect_index_obj_dst(self):
        obj_dst_id_list = []
        obj_dst_index_arr = []
        for env_id in range(self.num_envs):
            _curr_obj_index_store = self.env.obj_index_store[env_id]
            obj_dst_id = self.env.init_asset["env_info"][env_id]["task_info"]["obj_dst_id"]
            obj_dst_id_list.append(obj_dst_id)
            obj_dst_handle = _curr_obj_index_store[obj_dst_id]
            obj_dst_index_arr.append(obj_dst_handle)
        self.obj_dst_id_list = obj_dst_id_list
        self.obj_dst_index_arr = torch.as_tensor(obj_dst_index_arr, dtype=torch.long, device=self.device)

    def _load_obj_dst_interior(self):
        self.obj_dst_interior_store = {}
        for env_id in range(self.num_envs):
            obj_dst_id = self.env.init_asset["env_info"][env_id]["task_info"]["obj_dst_id"]
            if obj_dst_id in self.obj_dst_interior_store:
                continue
            obj_dst_interior = self.env.init_asset["obj_interior"][obj_dst_id]
            obj_dst_interior_verts = torch.as_tensor(obj_dst_interior.vertices, dtype=self.dtype, device=self.device)
            obj_dst_interior_faces = torch.as_tensor(obj_dst_interior.faces, dtype=torch.long, device=self.device)
            self.obj_dst_interior_store[obj_dst_id] = {"verts": obj_dst_interior_verts, "faces": obj_dst_interior_faces}

    def attach_env(self, env: EnvGeneral):
        super().attach_env(env)
        assert self.num_envs == self.env.num_envs

        # region: index
        self._collect_index_rh()

        if self.enable_evaluation:
            self._collect_index_fluid()
            self._collect_index_obj_dst()
            self._load_obj_dst_interior()
        # endregion: index

        # region: actuation
        self.dt = self.env.dt
        self.num_dof = self.rh_dof_index_arr.shape[1]
        self.target_prev = torch.zeros((self.num_envs, self.num_dof), dtype=self.dtype, device=self.device)
        self.target_curr = torch.zeros((self.num_envs, self.num_dof), dtype=self.dtype, device=self.device)
        self.target_dof_pos = torch.zeros_like(self.env.dof_pos)
        self.apply_force = torch.zeros((self.env.rigid_body_state.shape[0], 3), dtype=self.dtype, device=self.device)
        self.apply_torque = torch.zeros((self.env.rigid_body_state.shape[0], 3), dtype=self.dtype, device=self.device)

        self.actuated_dof_indices = torch.as_tensor(
            self.env.meta_info["rh"]["actuated_dof_index"], dtype=torch.long, device=self.device
        )
        self.num_shadow_hand_dofs = self.env.limit_info["rh"]["default_pos"].shape[0]
        self.shadow_hand_dof_lower_limits = torch.as_tensor(
            self.env.limit_info["rh"]["lower"], dtype=self.dtype, device=self.device
        )
        self.shadow_hand_dof_upper_limits = torch.as_tensor(
            self.env.limit_info["rh"]["upper"], dtype=self.dtype, device=self.device
        )
        # endregion

        # region: task-related
        ## hand_init
        hand_init_transf, hand_init_tsl, hand_init_quat, hand_init_dof_pos = [], [], [], []
        for env_id in range(self.num_envs):
            _init_tsl = self.env.init_asset["env_info"][env_id]["rh_info"]["tsl"]
            _init_quat = self.env.init_asset["env_info"][env_id]["rh_info"]["quat"]
            _init_dof_pos = self.env.init_asset["env_info"][env_id]["rh_info"]["dof_pos"]
            _init_rotmat = quat_to_rotmat_np(_init_quat)
            _init_transf = assemble_T_np(_init_tsl, _init_rotmat)
            hand_init_transf.append(_init_transf)
            hand_init_tsl.append(_init_tsl)
            hand_init_quat.append(_init_quat)
            hand_init_dof_pos.append(_init_dof_pos)
        self.init_rh_palm_tsl = torch.as_tensor(np.stack(hand_init_tsl, axis=0), dtype=self.dtype, device=self.device)
        self.init_rh_palm_quat = torch.as_tensor(np.stack(hand_init_quat, axis=0), dtype=self.dtype, device=self.device)
        self.init_rh_dof_pos = torch.as_tensor(
            np.stack(hand_init_dof_pos, axis=0), dtype=self.dtype, device=self.device
        )
        # endregion

        # region: control rollout
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        # endregion

        # region: set env attr
        # set initial action for dof_pos
        _action = torch.zeros((self.num_envs, self.act_dim), dtype=self.dtype, device=self.device)
        _action[:, :6] = 0.0
        _action[:, 6:] = unscale(
            self.init_rh_dof_pos[:, self.actuated_dof_indices],
            self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
            self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
        )
        self.env.actions = _action

        # clip
        self.env.clip_actions = 1.0
        self.env.clip_obs = 5.0

        # endregion

    def detach_env(self):
        super().detach_env()
        self.clear_buf()

    # reset_idx
    def reset_idx(self, env_id_to_reset: torch.Tensor):
        self.successes[env_id_to_reset] = 0

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

        self.apply_force[self.rh_palm_rb_index_arr, :] = (
            action[:, 0:3] * self.dt * self.translation_scale * 100000
        )  # ORI: 100000
        self.apply_torque[self.rh_palm_rb_index_arr, :] = (
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

        # do not modify raw action
        return action

    def compute_observation(self):
        obs = torch.zeros((self.num_envs, self.obs_dim), dtype=self.dtype, device=self.device)
        
        # region: evaluation
        if self.enable_evaluation:
            self.fluid_particle_tsl_list = []
            for fluid_index_arr in self.fluid_particle_index_arr_list:
                fluid_tsl = self.env.root_state[fluid_index_arr, 0:3]
                self.fluid_particle_tsl_list.append(fluid_tsl)
            self.obj_dst_tsl = self.env.root_state[self.obj_dst_index_arr, 0:3]
            self.obj_dst_quat = self.env.root_state[self.obj_dst_index_arr, 3:7][:, [3, 0, 1, 2]]
        # endregion

        self.env.obs_buf[:, :] = obs

    def compute_reward(self, action: torch.Tensor):
        # handle reset
        resets = self.env.reset_buf
        resets = torch.where(self.env.progress_buf >= self.env.max_episode_length, torch.ones_like(resets), resets)
        successes = self.successes
        current_successes = self.current_successes

        # evaluation
        if self.enable_evaluation:
            # overwrite success
            successes = []
            for env_id in range(self.num_envs):
                fluid_tsl = self.fluid_particle_tsl_list[env_id]
                obj_dst_id = self.obj_dst_id_list[env_id]
                container_dst_interior_verts = self.obj_dst_interior_store[obj_dst_id]["verts"]
                container_dst_interior_faces = self.obj_dst_interior_store[obj_dst_id]["faces"]
                container_dst_interior_verts = (
                    quat_apply(self.obj_dst_quat[env_id], container_dst_interior_verts) + self.obj_dst_tsl[env_id]
                )
                fluid_inside_dst = check_sign(
                    container_dst_interior_verts.unsqueeze(0), container_dst_interior_faces, fluid_tsl.unsqueeze(0)
                ).squeeze(0)
                fluid_inside_ratio = float(torch.mean(fluid_inside_dst.float()))
                successes.append(fluid_inside_ratio >= self.evaluation_success_threshold)
            successes = torch.as_tensor(successes, dtype=torch.float, device=self.device)

            # override existing
            current_successes = torch.where(resets > 0, successes, current_successes)

        # handle buf
        self.env.reset_buf[:] = resets
        self.env.rew_buf[:] = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self.successes[:] = successes
        self.current_successes[:] = current_successes

    def viz_debug(self):
        self.env.gym.clear_lines(self.env.viewer)
