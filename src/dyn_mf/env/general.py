from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing import Any, Optional, Callable

import sys
import os
import math
import numpy as np
import pickle
import logging
from dev_fn.util.console_io import RedirectStream, pformat, pprint
from dev_fn.transform.rotation_np import rotmat_to_quat_np, quat_to_euler_angle_np
from dev_fn.dataset import segment

with RedirectStream(), RedirectStream(sys.stderr):
    from isaacgym import gymapi, gymutil, gymtorch
    from isaacgymenvs.tasks.base.vec_task import VecTask
    from isaacgymenvs.utils.torch_jit_utils import to_torch
import torch
import itertools

from . import const as env_const
from . import tool as env_tool
from ..scene import FrameConvention, GeomCata
from .common import EnvCallbackContext


_logger = logging.getLogger(__name__)

# TODO list:
# 4. validate the ppo algorithm in artigrasp repo
# 4.1 set goals
# 4.2 obs
# 4.3 reward


def _extract_init_info(init_asset):
    init_info = {
        "obj_aux": [],
        "obj": [],
        "lh": [] if init_asset["lh_def"] is not None else None,
        "rh": [] if init_asset["rh_def"] is not None else None,
    }

    def _handle_obj(_info):
        obj_id = _info["id"]
        transf = _info["transf"]
        tsl = transf[:3, 3]
        rotmat = transf[:3, :3]
        quat = rotmat_to_quat_np(rotmat)
        dof_pos = _info["dof_pos"]
        return {
            "id": obj_id,
            "transf": transf,
            "tsl": tsl,
            "quat": quat,
            "dof_pos": dof_pos,
        }

    env_info_list = init_asset["env_info"]
    num_env = len(env_info_list)
    for env_id, env_info in enumerate(env_info_list):
        # obj_aux
        obj_info_aux_list = env_info["obj_info_aux"]
        _obj_aux_store = {}
        for obj_info_aux in obj_info_aux_list:
            obj_aux_id = obj_info_aux["id"]
            _obj_aux_store[obj_aux_id] = _handle_obj(obj_info_aux)
        init_info["obj_aux"].append(_obj_aux_store)

        # obj
        obj_info_list = env_info["obj_info"]
        _obj_store = {}
        for obj_info in obj_info_list:
            obj_id = obj_info["id"]
            _obj_store[obj_id] = _handle_obj(obj_info)
        init_info["obj"].append(_obj_store)

        # lh
        if init_asset["lh_def"] is not None:
            lh_info = env_info["lh_info"]
            init_info["lh"].append(
                {
                    "tsl": lh_info["tsl"],
                    "quat": lh_info["quat"],
                    "dof_pos": lh_info["dof_pos"],
                }
            )

        # rh
        if init_asset["rh_def"] is not None:
            rh_info = env_info["rh_info"]
            init_info["rh"].append(
                {
                    "tsl": rh_info["tsl"],
                    "quat": rh_info["quat"],
                    "dof_pos": rh_info["dof_pos"],
                }
            )

    return init_info


def _dof_count_to_index(dof_count):
    res = {}
    front = 0
    for k, v in dof_count.items():
        res[k] = np.array(list(range(front, front + v)), dtype=np.int64)
        front += v
    return res


def _rb_count_to_index(rb_count, rb_index_map):
    res, res_index_map = {}, {}
    front = 0
    for k, v in rb_count.items():
        res[k] = np.array(list(range(front, front + v)), dtype=np.int64)
        res_index_map[k] = {kk: vv + front for kk, vv in rb_index_map[k].items()}
        front += v
    return res, res_index_map


class EnvGeneral(VecTask):
    def __init__(
        self,
        asset_loader: Callable,
        isaacgym_cfg: dict[str, Any],
        rl_device: str,
        sim_device: str,
        graphics_device_id: int,
        headless: bool,
        virtual_screen_capture: bool = False,
        force_render: bool = False,
        env_callback_ctx: Optional[EnvCallbackContext] = None,
        override_cfg: bool = True,
    ):
        # asset loader
        self.asset_loader = asset_loader
        ## asset
        self.init_asset = self.asset_loader()
        ## determine up axis
        self.up_axis = "z" if self.asset_loader.frame_convention == FrameConvention.Z_UP else "y"
        self.gravity = [0.0, 0.0, -9.81] if self.up_axis == "z" else [0.0, -9.81, 0.0]

        # callback
        self.env_callback_ctx = env_callback_ctx

        # init VecTask
        self.debug_viz = isaacgym_cfg.get("env", {}).get("enableDebugVis", False)
        # maintainence code
        self.cfg = isaacgym_cfg.copy()  # VecTask required
        # populate self.cfg
        if override_cfg:
            use_gpu_pipeline = not (sim_device == "cpu")
            self.cfg["physics_engine"] = "physx"
            self.cfg["sim"] = dict(
                dt=1.0 / 60,
                substeps=4,
                up_axis=self.up_axis,
                use_gpu_pipeline=use_gpu_pipeline,
                gravity=self.gravity,
                controlFrequencyInv=2,
            )
            self.cfg["sim"]["phsyx"] = dict(
                solver_type=1,
                contact_offset=0.002,
                rest_offset=0.00000,
                bounce_threshold_velocity=0.2,
                max_depenetration_velocity=1000,
                friction_offset_threshold=0.008,
                friction_correlation_distance=0.005,
                num_position_iterations=6,
                num_velocity_iterations=1,
                num_threads=4,
                use_gpu=use_gpu_pipeline,
                max_gpu_contact_pairs=1000 * 1000 * 10,
                default_buffer_size_multiplier=10,
            )
        _logger.info("env using dt %f", self.cfg["sim"]["dt"])
        max_episode_length = env_callback_ctx.timeout // (
            self.cfg["sim"]["dt"] * self.cfg["sim"]["controlFrequencyInv"]
        )
        _logger.info("env max_episode_length %d", max_episode_length)
        self.cfg["env"].update(
            dict(
                numObservations=env_callback_ctx.obs_dim,
                numActions=env_callback_ctx.act_dim,
                max_episode_length=max_episode_length,
            )
        )
        ## max episode
        self.max_episode_length = self.cfg["env"]["max_episode_length"]  # TODO: compute use time
        ## sim_device
        self.sim_device = sim_device  # alternate name of self.device

        # attr for env creation
        ## viz
        self.hand_color = [147 / 255, 215 / 255, 160 / 255]
        self.object_color = [90 / 255, 94 / 255, 173 / 255]
        self.object_color_2 = [150 / 255, 150 / 255, 150 / 255]

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        assert self.num_envs == self.cfg["env"]["numEnvs"]
        if self.cfg["sim"]["use_gpu_pipeline"]:
            assert self.device == self.sim_device  # sanity check

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.root_state_init = self.root_state.detach().clone()

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(-1, 2)
        self.dof_pos = self.dof_state[..., 0:1]
        self.dof_vel = self.dof_state[..., 1:2]
        self.dof_force = gymtorch.wrap_tensor(dof_force_tensor).view(-1, 1)
        self.dof_state_init = self.dof_state.detach().clone()

        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor).view(-1, 13)
        self.net_contact_force = gymtorch.wrap_tensor(net_contact_force_tensor).view(-1, 3)

        self.total_dofs = self.gym.get_sim_dof_count(self.sim)  # total dofs
        assert self.total_dofs == sum(self.dof_count_list)

        # Fh env to context
        if self.env_callback_ctx is not None:
            self.env_callback_ctx.attach_env(self)

    def create_sim(self):
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        # load plane
        plane_params = gymapi.PlaneParams()
        plane_params.distance = -env_const.GROUND_OFFSET
        if self.up_axis == "z":
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        else:
            plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

        # load asset
        # {
        #  "env_info": [{
        #   "process_key":
        #   "frame_id":
        #   "obj_info": [{"transf", "geom_cata", "geom_param", "dof_pos", "static", "collision_filter", "vhacd"}, ...]
        #   "obj_aux_info": [{"transf", "geom_cata", "geom_param", "dof_pos", "static", "collision_filter", "vhacd"}, ...] # (table, stage, etc...)
        #   "lh_info": {"tsl", "quat", "dof_pos", "collision_filter"}
        #   "rh_info": {"tsl", "quat", "dof_pos", "collision_filter"}
        #   },...],
        #  "obj_def": {obj_id: {"geom_cata", "geom_param", "static"}, ...},
        #  "obj_aux_def": {obj_aux_id: {"geom_cata", "geom_param", "static"}, ...},
        #  "lh_def": .....,
        #  "rh_def": .....,
        # }
        self.meta_info = {}
        self.limit_info = {}
        asset_store_obj_aux = {}
        asset_store_obj = {}
        # handle obj_aux
        obj_aux_def = self.init_asset["obj_aux_def"]
        for obj_aux_id, obj_def_item in obj_aux_def.items():
            current_asset = self._create_asset(obj_def_item)
            asset_store_obj_aux[obj_aux_id] = current_asset
            self.meta_info[obj_aux_id] = {
                "num_body": self.gym.get_asset_rigid_body_count(current_asset),
                "num_shape": self.gym.get_asset_rigid_shape_count(current_asset),
                "num_dof": self.gym.get_asset_dof_count(current_asset),
                "num_actuator": self.gym.get_asset_actuator_count(current_asset),
                "num_tendon": self.gym.get_asset_tendon_count(current_asset),
            }

        # handle obj
        obj_def = self.init_asset["obj_def"]
        for obj_id, obj_def_item in obj_def.items():
            current_asset = self._create_asset(obj_def_item)
            asset_store_obj[obj_id] = current_asset
            self.meta_info[obj_id] = {
                "num_body": self.gym.get_asset_rigid_body_count(current_asset),
                "num_shape": self.gym.get_asset_rigid_shape_count(current_asset),
                "num_dof": self.gym.get_asset_dof_count(current_asset),
                "num_actuator": self.gym.get_asset_actuator_count(current_asset),
                "num_tendon": self.gym.get_asset_tendon_count(current_asset),
            }

        # load agents
        # TODO: general multi-agent
        self.lh_enabled, self.rh_enabled = self.init_asset["lh_def"] is not None, self.init_asset["rh_def"] is not None
        self.num_agents = int(self.lh_enabled) + int(self.rh_enabled)
        if self.lh_enabled:
            lh_asset_options = gymapi.AssetOptions()
            lh_asset_options.flip_visual_attachments = False
            lh_asset_options.fix_base_link = self.init_asset["lh_def"]["static"]
            lh_asset_options.disable_gravity = True
            lh_asset_options.collapse_fixed_joints = False
            lh_asset_options.thickness = self.init_asset["lh_def"].get("thickness", 0.001)
            if self.init_asset[f"lh_def"].get("cata", None) == "shadowhand_no_forearm":
                lh_asset_options.angular_damping = 100
                lh_asset_options.linear_damping = 100
                lh_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            geom_param_lh = self.init_asset["lh_def"]["geom_param"]
            asset_lh = self.gym.load_asset(
                self.sim, geom_param_lh["asset_dir"], geom_param_lh["asset_name"], lh_asset_options
            )
            self.meta_info["lh"] = {
                "num_body": self.gym.get_asset_rigid_body_count(asset_lh),
                "num_shape": self.gym.get_asset_rigid_shape_count(asset_lh),
                "num_dof": self.gym.get_asset_dof_count(asset_lh),
                "num_actuator": self.gym.get_asset_actuator_count(asset_lh),
                "num_tendon": self.gym.get_asset_tendon_count(asset_lh),
            }

            # tendon
            if self.init_asset[f"lh_def"].get("cata", None) == "shadowhand_no_forearm":
                limit_stiffness = 30
                t_damping = 0.1
                relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
                tendon_props = self.gym.get_asset_tendon_properties(asset_lh)
                for i in range(self.meta_info["lh"]["num_tendon"]):
                    for rt in relevant_tendons:
                        if self.gym.get_asset_tendon_name(asset_lh, i) == rt:
                            tendon_props[i].limit_stiffness = limit_stiffness
                            tendon_props[i].damping = t_damping
                self.gym.set_asset_tendon_properties(asset_lh, tendon_props)

            # dof actutation
            actuated_dof_name_lh = [
                self.gym.get_asset_actuator_joint_name(asset_lh, i) for i in range(self.meta_info["lh"]["num_actuator"])
            ]
            actuated_dof_index_lh = [self.gym.find_asset_dof_index(asset_lh, name) for name in actuated_dof_name_lh]
            self.meta_info["lh"]["actuated_dof_name"] = actuated_dof_name_lh
            self.meta_info["lh"]["actuated_dof_index"] = actuated_dof_index_lh

            # dof
            asset_lh_dof_props = self.gym.get_asset_dof_properties(asset_lh)
            self.limit_info["lh"] = {
                "lower": np.asarray(asset_lh_dof_props["lower"]).copy().astype(np.float32),
                "upper": np.asarray(asset_lh_dof_props["upper"]).copy().astype(np.float32),
            }
            self.limit_info["lh"]["default_pos"] = np.zeros_like(self.limit_info["lh"]["lower"])
            self.limit_info["lh"]["default_vel"] = np.zeros_like(self.limit_info["lh"]["lower"])

        if self.rh_enabled:
            rh_asset_options = gymapi.AssetOptions()
            rh_asset_options.flip_visual_attachments = False
            rh_asset_options.fix_base_link = self.init_asset["rh_def"]["static"]
            rh_asset_options.disable_gravity = True
            rh_asset_options.collapse_fixed_joints = False
            rh_asset_options.thickness = self.init_asset["rh_def"].get("thickness", 0.001)
            if self.init_asset[f"rh_def"].get("cata", None) == "shadowhand_no_forearm":
                rh_asset_options.angular_damping = 100
                rh_asset_options.linear_damping = 100
                rh_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            geom_param_rh = self.init_asset["rh_def"]["geom_param"]
            asset_rh = self.gym.load_asset(
                self.sim, geom_param_rh["asset_dir"], geom_param_rh["asset_name"], rh_asset_options
            )
            self.meta_info["rh"] = {
                "num_body": self.gym.get_asset_rigid_body_count(asset_rh),
                "num_shape": self.gym.get_asset_rigid_shape_count(asset_rh),
                "num_dof": self.gym.get_asset_dof_count(asset_rh),
                "num_actuator": self.gym.get_asset_actuator_count(asset_rh),
                "num_tendon": self.gym.get_asset_tendon_count(asset_rh),
            }

            # tendon
            if self.init_asset[f"rh_def"].get("cata", None) == "shadowhand_no_forearm":
                limit_stiffness = 30
                t_damping = 0.1
                relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
                tendon_props = self.gym.get_asset_tendon_properties(asset_rh)
                for i in range(self.meta_info["rh"]["num_tendon"]):
                    for rt in relevant_tendons:
                        if self.gym.get_asset_tendon_name(asset_rh, i) == rt:
                            tendon_props[i].limit_stiffness = limit_stiffness
                            tendon_props[i].damping = t_damping
                self.gym.set_asset_tendon_properties(asset_rh, tendon_props)

            # dof actutation
            actuated_dof_name_rh = [
                self.gym.get_asset_actuator_joint_name(asset_rh, i) for i in range(self.meta_info["rh"]["num_actuator"])
            ]
            actuated_dof_index_rh = [self.gym.find_asset_dof_index(asset_rh, name) for name in actuated_dof_name_rh]
            self.meta_info["rh"]["actuated_dof_name"] = actuated_dof_name_rh
            self.meta_info["rh"]["actuated_dof_index"] = actuated_dof_index_rh

            # dof
            asset_rh_dof_props = self.gym.get_asset_dof_properties(asset_rh)
            self.limit_info["rh"] = {
                "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
                "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
            }
            self.limit_info["rh"]["default_pos"] = np.zeros_like(self.limit_info["rh"]["lower"])
            self.limit_info["rh"]["default_vel"] = np.zeros_like(self.limit_info["rh"]["lower"])

        _logger.info("meta_info: \n%s" % pformat(self.meta_info))
        _logger.info("limit_info: \n%s" % pformat(self.limit_info))

        # load envs
        if self.up_axis == "z":
            spacing = 2.0  # TODO
            lower = gymapi.Vec3(-spacing, -spacing, -0.05)
            upper = gymapi.Vec3(spacing, spacing, 2.0)
        else:
            spacing = 2.0  # TODO
            lower = gymapi.Vec3(-spacing, -0.05, -spacing)
            upper = gymapi.Vec3(spacing, 2.0, spacing)

        num_per_row = math.ceil(np.sqrt(self.num_envs))
        self.env_list = []
        self.lh_actor_list = []
        self.rh_actor_list = []
        self.lh_index_arr = [] if self.lh_enabled else None
        self.rh_index_arr = [] if self.rh_enabled else None

        self.obj_aux_actor_store = []
        self.obj_actor_store = []
        self.obj_aux_index_store = []  # variable length
        self.obj_index_store = []  # variable length
        # self.hand_index_store = []  # all basic embodiment should share across env

        self.env_dof_count_store = []
        self.env_dof_index_store = []  # variable length
        self.dof_count_list = []
        self.dof_index_begin_list = []

        self.env_rb_count_store = []
        self.env_rb_index_store = []
        self.env_rb_index_map_store = []  # allow index by link name
        self.rb_count_list = []
        self.rb_index_begin_list = []

        dof_front = 0
        rb_front = 0
        for env_id in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # get handle of current env asset
            env_asset = self.init_asset["env_info"][env_id]
            _curr_dof_count = {}
            _curr_rb_count, _curr_rb_index_map = {}, {}

            # obj_aux
            _curr_obj_aux_actor = {}
            _curr_obj_aux_index = {}
            for obj_aux_info in env_asset["obj_info_aux"]:
                obj_aux_actor, obj_aux_index = self._create_obj_actor(
                    env_ptr, env_id, obj_aux_info, asset_store_obj_aux
                )
                obj_aux_dof_count = self.gym.get_actor_dof_count(env_ptr, obj_aux_actor)
                _curr_obj_aux_actor[obj_aux_info["id"]] = obj_aux_actor
                _curr_obj_aux_index[obj_aux_info["id"]] = obj_aux_index
                if obj_aux_dof_count > 0:
                    _curr_dof_count[obj_aux_info["id"]] = obj_aux_dof_count
                obj_aux_rb_count = self.gym.get_actor_rigid_body_count(env_ptr, obj_aux_actor)
                obj_aux_rb_dict = self.gym.get_actor_rigid_body_dict(env_ptr, obj_aux_actor)
                _curr_rb_count[obj_aux_info["id"]] = obj_aux_rb_count
                _curr_rb_index_map[obj_aux_info["id"]] = dict(sorted(obj_aux_rb_dict.items(), key=lambda item: item[1]))
            self.obj_aux_actor_store.append(_curr_obj_aux_actor)
            self.obj_aux_index_store.append(_curr_obj_aux_index)

            # obj
            _curr_obj_actor = {}
            _curr_obj_index = {}
            for obj_info in env_asset["obj_info"]:
                obj_actor, obj_index = self._create_obj_actor(env_ptr, env_id, obj_info, asset_store_obj)
                self.gym.set_rigid_body_color(env_ptr, obj_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*self.object_color))
                obj_dof_count = self.gym.get_actor_dof_count(env_ptr, obj_actor)
                _curr_obj_actor[obj_info["id"]] = obj_actor
                _curr_obj_index[obj_info["id"]] = obj_index
                if obj_dof_count > 0:
                    _curr_dof_count[obj_info["id"]] = obj_dof_count
                obj_rb_count = self.gym.get_actor_rigid_body_count(env_ptr, obj_actor)
                obj_rb_dict = self.gym.get_actor_rigid_body_dict(env_ptr, obj_actor)
                _curr_rb_count[obj_info["id"]] = obj_rb_count
                _curr_rb_index_map[obj_info["id"]] = dict(sorted(obj_rb_dict.items(), key=lambda item: item[1]))
            self.obj_actor_store.append(_curr_obj_actor)
            self.obj_index_store.append(_curr_obj_index)

            # agent
            stiffness = 1.00  # TODO: move to const
            damping = 0.10
            hand_friction = 1.0  # TODO: This is the default value; need way to specify this value in asset creation
            obj_friction = 1.0
            if self.lh_enabled:
                lh_info = env_asset["lh_info"]
                lh_tsl = lh_info["tsl"]
                lh_quat = lh_info["quat"]
                lh_actor = self.gym.create_actor(
                    env_ptr,
                    asset_lh,
                    gymapi.Transform(
                        p=gymapi.Vec3(lh_tsl[0], lh_tsl[1], lh_tsl[2]),
                        r=gymapi.Quat(lh_quat[1], lh_quat[2], lh_quat[3], lh_quat[0]),
                    ),
                    "lh",
                    env_id,
                    env_asset["lh_info"]["collision_filter"],
                )
                lh_dof_props = self.gym.get_actor_dof_properties(env_ptr, lh_actor)
                if self.init_asset[f"lh_def"].get("cata", None) == "mano":
                    lh_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
                    lh_dof_props["stiffness"].fill(stiffness)
                    lh_dof_props["damping"].fill(damping)
                elif self.init_asset[f"lh_def"].get("cata", None) == "shadowhand_no_forearm":
                    _lh_drive = (lh_dof_props["driveMode"] == gymapi.DOF_MODE_POS)
                    lh_dof_props["stiffness"][_lh_drive] *= 10
                    lh_dof_props["effort"][_lh_drive] *= 10
                self.gym.set_actor_dof_properties(env_ptr, lh_actor, lh_dof_props)
                for _o in range(self.meta_info["lh"]["num_body"]):
                    self.gym.set_rigid_body_color(
                        env_ptr, lh_actor, _o, gymapi.MESH_VISUAL, gymapi.Vec3(*self.hand_color)
                    )

                lh_index = self.gym.get_actor_index(env_ptr, lh_actor, gymapi.DOMAIN_SIM)
                lh_dof_count = self.gym.get_actor_dof_count(env_ptr, lh_actor)
                _curr_dof_count["lh"] = lh_dof_count
                lh_rb_count = self.gym.get_actor_rigid_body_count(env_ptr, lh_actor)
                lh_rb_dict = self.gym.get_actor_rigid_body_dict(env_ptr, lh_actor)
                _curr_rb_count["lh"] = lh_rb_count
                _curr_rb_index_map["lh"] = dict(sorted(lh_rb_dict.items(), key=lambda item: item[1]))
                self.lh_actor_list.append(lh_actor)
                self.lh_index_arr.append(lh_index)

            if self.rh_enabled:
                rh_info = env_asset["rh_info"]
                rh_tsl = rh_info["tsl"]
                rh_quat = rh_info["quat"]
                rh_actor = self.gym.create_actor(
                    env_ptr,
                    asset_rh,
                    gymapi.Transform(
                        p=gymapi.Vec3(rh_tsl[0], rh_tsl[1], rh_tsl[2]),
                        r=gymapi.Quat(rh_quat[1], rh_quat[2], rh_quat[3], rh_quat[0]),
                    ),
                    "rh",
                    env_id,
                    env_asset["rh_info"]["collision_filter"],
                )
                rh_dof_props = self.gym.get_actor_dof_properties(env_ptr, rh_actor)
                if self.init_asset[f"rh_def"].get("cata", None) == "mano":
                    rh_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
                    rh_dof_props["stiffness"].fill(stiffness)
                    rh_dof_props["damping"].fill(damping)
                elif self.init_asset[f"rh_def"].get("cata", None) == "shadowhand_no_forearm":
                    _rh_drive = (rh_dof_props["driveMode"] == gymapi.DOF_MODE_POS)
                    rh_dof_props["stiffness"][_rh_drive] *= 10
                    rh_dof_props["effort"][_rh_drive] *= 10
                self.gym.set_actor_dof_properties(env_ptr, rh_actor, rh_dof_props)
                for _o in range(self.meta_info["rh"]["num_body"]):
                    self.gym.set_rigid_body_color(
                        env_ptr, rh_actor, _o, gymapi.MESH_VISUAL, gymapi.Vec3(*self.hand_color)
                    )

                rh_index = self.gym.get_actor_index(env_ptr, rh_actor, gymapi.DOMAIN_SIM)
                rh_dof_count = self.gym.get_actor_dof_count(env_ptr, rh_actor)
                _curr_dof_count["rh"] = rh_dof_count
                rh_rb_count = self.gym.get_actor_rigid_body_count(env_ptr, rh_actor)
                rh_rb_dict = self.gym.get_actor_rigid_body_dict(env_ptr, rh_actor)
                _curr_rb_count["rh"] = rh_rb_count
                _curr_rb_index_map["rh"] = dict(sorted(rh_rb_dict.items(), key=lambda item: item[1]))
                self.rh_actor_list.append(rh_actor)
                self.rh_index_arr.append(rh_index)

            _curr_dof_index = _dof_count_to_index(_curr_dof_count)
            _curr_dof_num = sum(_curr_dof_count.values())
            self.env_dof_count_store.append(_curr_dof_count)
            self.env_dof_index_store.append(_curr_dof_index)
            self.dof_count_list.append(_curr_dof_num)
            self.dof_index_begin_list.append(dof_front)
            dof_front += _curr_dof_num

            _curr_rb_index, _curr_rb_index_map = _rb_count_to_index(_curr_rb_count, _curr_rb_index_map)
            _curr_rb_num = sum(_curr_rb_count.values())
            self.env_rb_count_store.append(_curr_rb_count)
            self.env_rb_index_store.append(_curr_rb_index)
            self.env_rb_index_map_store.append(_curr_rb_index_map)
            self.rb_count_list.append(_curr_rb_num)
            self.rb_index_begin_list.append(rb_front)
            rb_front += _curr_rb_num

            # record env
            self.env_list.append(env_ptr)

        self.lh_index_arr = (
            to_torch(self.lh_index_arr, dtype=torch.long, device=self.device) if self.lh_enabled else None
        )
        self.rh_index_arr = (
            to_torch(self.rh_index_arr, dtype=torch.long, device=self.device) if self.rh_enabled else None
        )

        # extract init info
        self.init_info = _extract_init_info(self.init_asset)

    def _create_asset(self, obj_def_item):
        asset_options = gymapi.AssetOptions()
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.density = obj_def_item.get("density", 990)
        asset_options.thickness = obj_def_item.get("thickness", 0.001)
        if obj_def_item["static"]:
            asset_options.fix_base_link = True
        if obj_def_item.get("vhacd", False):
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 300000
        if obj_def_item.get("submesh", False):
            asset_options.convex_decomposition_from_submeshes = True
        if obj_def_item.get("collapse_fixed_joints", False):
            asset_options.collapse_fixed_joints = True

        if obj_def_item["geom_cata"] == GeomCata.BOX:
            _w, _h, _d = obj_def_item["geom_param"]["dim"]
            current_asset = self.gym.create_box(self.sim, _w, _h, _d, asset_options)
        elif obj_def_item["geom_cata"] == GeomCata.URDF:
            _dir = obj_def_item["geom_param"]["asset_dir"]
            _name = obj_def_item["geom_param"]["asset_name"]
            current_asset = self.gym.load_asset(self.sim, _dir, _name, asset_options)

        return current_asset

    def _create_obj_actor(self, env_ptr, env_id, obj_info, asset_store):
        obj_dof_damping = 0.10  # TODO
        obj_dof_stiffness = 0.00  # TODO
        obj_dof_friction = 0.0001
        obj_dof_armature = 0.0001

        obj_id = obj_info["id"]
        obj_cur_tf = obj_info["transf"]
        pose = gymapi.Transform()
        tsl = obj_cur_tf[:3, 3]
        pose.p = gymapi.Vec3(tsl[0], tsl[1], tsl[2])
        quat = rotmat_to_quat_np(obj_cur_tf[:3, :3])
        pose.r = gymapi.Quat(quat[1], quat[2], quat[3], quat[0])

        current_asset = asset_store[obj_id]
        obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, obj_id, env_id, obj_info["collision_filter"])
        obj_index = self.gym.get_actor_index(env_ptr, obj_actor, gymapi.DOMAIN_SIM)

        if obj_info["dof_pos"] is not None:
            obj_dof_props = self.gym.get_actor_dof_properties(env_ptr, obj_actor)
            obj_dof_props["damping"].fill(obj_dof_damping)
            obj_dof_props["stiffness"].fill(obj_dof_stiffness)
            obj_dof_props["friction"].fill(obj_dof_friction)
            obj_dof_props["armature"].fill(obj_dof_armature)
            self.gym.set_actor_dof_properties(env_ptr, obj_actor, obj_dof_props)

        return obj_actor, obj_index

    def set_viewer(self):
        super().set_viewer()

        # reset cam pos
        cam_pos = gymapi.Vec3(2.0, 2.0, 2.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def reset_idx(self, env_id_to_reset):
        # TODO: apply_randomizations

        # reset object
        # collect root_state arr
        root_state_index_list, root_state_tsl_list, root_state_quat_xyzw_list = [], [], []
        actor_index_list, actor_dof_index_list, actor_dof_pos_list = [], [], []
        for env_id in env_id_to_reset.detach().clone().tolist():
            # handle aux
            root_state_index_list.extend(self.obj_aux_index_store[env_id].values())
            _info_store = self.init_info["obj_aux"][env_id]
            root_state_tsl_list.extend(_info["tsl"] for _info in _info_store.values())
            root_state_quat_xyzw_list.extend(_info["quat"][[1, 2, 3, 0]] for _info in _info_store.values())

            # handle obj
            root_state_index_list.extend(self.obj_index_store[env_id].values())
            _info_store = self.init_info["obj"][env_id]
            root_state_tsl_list.extend(_info["tsl"] for _info in _info_store.values())
            root_state_quat_xyzw_list.extend(_info["quat"][[1, 2, 3, 0]] for _info in _info_store.values())

            # handle hand
            for hand_side in segment.HAND_SIDE:
                if self.init_info[hand_side] is None:
                    continue
                _index_arr = self.lh_index_arr if hand_side == "lh" else self.rh_index_arr
                root_state_index_list.append(int(_index_arr[env_id]))
                _info_store = self.init_info[hand_side][env_id]
                root_state_tsl_list.append(_info_store["tsl"])
                root_state_quat_xyzw_list.append(_info_store["quat"][[1, 2, 3, 0]])

            local_dof_index_begin = self.dof_index_begin_list[env_id]
            local_dof_index_store = self.env_dof_index_store[env_id]
            # handle aux dof
            for obj_aux_id, obj_aux_info in self.init_info["obj_aux"][env_id].items():
                if obj_aux_info["dof_pos"] is None:
                    continue
                full_dof_index = (local_dof_index_begin + local_dof_index_store[obj_aux_id]).tolist()
                full_dof_pos = torch.as_tensor(obj_aux_info["dof_pos"], dtype=torch.float32, device=self.device)
                actor_index_list.append(self.obj_aux_index_store[env_id][obj_aux_id])
                actor_dof_index_list.extend(full_dof_index)
                actor_dof_pos_list.append(full_dof_pos)

            # handle obj dof
            for obj_id, obj_info in self.init_info["obj"][env_id].items():
                if obj_info["dof_pos"] is None:
                    continue
                full_dof_index = (local_dof_index_begin + local_dof_index_store[obj_id]).tolist()
                full_dof_pos = torch.as_tensor(obj_info["dof_pos"], dtype=torch.float32, device=self.device)
                actor_index_list.append(self.obj_index_store[env_id][obj_id])
                actor_dof_index_list.extend(full_dof_index)
                actor_dof_pos_list.append(full_dof_pos)

            # handle hand
            for hand_side in segment.HAND_SIDE:
                if self.init_info[hand_side] is None:
                    continue
                full_dof_index = (local_dof_index_begin + local_dof_index_store[hand_side]).tolist()
                if self.init_asset[f"{hand_side}_def"].get("cata", None) == "mano":
                    full_dof_pos = env_tool.assemble_full_dof_pos(
                        self.init_info[hand_side][env_id],
                        zero_root=True,
                        dtype=torch.float32,
                        device=self.device,
                    )
                else:
                    full_dof_pos = torch.as_tensor(
                        self.init_info[hand_side][env_id]["dof_pos"], dtype=torch.float32, device=self.device
                    )
                _index_arr = self.lh_index_arr if hand_side == "lh" else self.rh_index_arr
                actor_index_list.append(int(_index_arr[env_id]))
                actor_dof_index_list.extend(full_dof_index)
                actor_dof_pos_list.append(full_dof_pos)

        ## root_state
        obj_index_flat = torch.as_tensor(
            np.array(root_state_index_list, dtype=np.int64), dtype=torch.long, device=self.device
        )
        tsl_arr_flat = torch.as_tensor(np.stack(root_state_tsl_list, axis=0), dtype=torch.float32, device=self.device)
        quat_xyzw_arr_flat = torch.as_tensor(
            np.stack(root_state_quat_xyzw_list, axis=0), dtype=torch.float32, device=self.device
        )
        self.root_state[obj_index_flat, 0:3] = tsl_arr_flat
        self.root_state[obj_index_flat, 3:7] = quat_xyzw_arr_flat
        self.root_state[obj_index_flat, 7:13] = torch.zeros(
            (len(obj_index_flat), 6), dtype=torch.float32, device=self.device
        )
        # clear root_state_init too
        self.root_state_init[obj_index_flat, :] = self.root_state[obj_index_flat, :].detach().clone()

        ## dof_pos
        actor_index_arr = torch.as_tensor(
            np.array(actor_index_list, dtype=np.int64), dtype=torch.long, device=self.device
        )
        actor_dof_index_flat = torch.as_tensor(
            np.array(actor_dof_index_list, dtype=np.int64), dtype=torch.long, device=self.device
        )
        actor_dof_pos_arr_flat = torch.cat(actor_dof_pos_list, dim=0).to(torch.float32)
        self.dof_pos[actor_dof_index_flat, :] = actor_dof_pos_arr_flat.unsqueeze(-1)
        self.dof_vel[actor_dof_index_flat, :] = 0.0
        # clear dof_state_init too
        self.dof_state_init[actor_dof_index_flat, :] = self.dof_state[actor_dof_index_flat, :].detach().clone()

        # apply
        _index = torch.cat((obj_index_flat,), dim=0).to(torch.int32)
        _n_index = len(_index)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(_index),
            _n_index,
        )
        _index = torch.cat((actor_index_arr,), dim=0).to(torch.int32)
        _n_index = len(_index)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(_index),
            _n_index,
        )

        # update buf
        self.progress_buf[env_id_to_reset] = 0
        self.reset_buf[env_id_to_reset] = 0

    def pre_physics_step(self, actions: torch.Tensor):
        env_id_to_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_id_to_reset) > 0:
            self.reset_idx(env_id_to_reset)

        self.actions = actions.clone().to(self.device)

        if self.env_callback_ctx is not None:
            self.actions = self.env_callback_ctx.apply_action(self.actions, env_id_to_reset)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.compute_observation()
        self.compute_reward(self.actions)

        # make sure to reset

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.env_callback_ctx.viz_debug()

    def compute_reward(self, actions):
        if self.env_callback_ctx is not None:
            self.env_callback_ctx.compute_reward(actions)

    def compute_observation(self):
        if self.env_callback_ctx is not None:
            self.env_callback_ctx.compute_observation()
