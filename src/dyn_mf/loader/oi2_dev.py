from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing import Optional, Mapping, Any

import os
import numpy as np
import torch
import pickle
import trimesh
import json
import open3d as o3d

import pytorch_kinematics
from dev_fn.util.console_io import pprint
from dev_fn.transform.rotation_np import rotmat_to_quat_np, euler_angle_to_quat_np
from dev_fn.transform.transform_np import assemble_T_np, inv_transf_np
from ..scene import FrameConvention, GeomCata
from .common import load_table_def, load_box_def, load_sphere_def
from .common import AssetLoader
from dev_fn.util.hash_util import hash_json_attr_map
from dev_fn.util.pbar_util import wrap_pbar
from ..dataset.oi2_dev import parse_task_code
from ..env.embodiment_meta import ShadowHand__NoForearm__Meta
from dev_fn.geo.ops.check_sign import check_sign


_thisdir = os.path.dirname(__file__)
_shadowhand_filedir = os.path.normpath(os.path.join(_thisdir, "../../..", "asset/shadowhand_no_forearm"))
_data_filedir = os.path.normpath(os.path.join(_thisdir, "../../..", "asset/oi2_dev"))


class OakInk2DevAssetLoader(AssetLoader):

    def __init__(
        self,
        num_env: int,
        task_info_list: list[Mapping[str, Any]],
        data_filedir: str = _data_filedir,
        shadowhand_filedir: str = _shadowhand_filedir,
        enable_evaluation: bool = False,
        fluid_particle_size: float = 0.006,
        fluid_particle_count: int = 125,
        verbose: bool = False,
    ) -> None:
        self.num_env = num_env
        self.data_filedir = data_filedir
        self.shadowhand_filedir = shadowhand_filedir
        self.task_info_list = task_info_list
        self.frame_convention = FrameConvention.Y_UP

        self.cache_dict = {}
        self.obj_model_store = {}

        # hand
        self.hand_type = "rh"

        # collision
        self.hand_collision_filter = 0b0000
        self.obj_collision_filter = 0b0000
        self.table_collision_filter = 0b0000

        # table
        # self.table_box_pos = (0, -0.02501, -0.025)
        # self.table_box_dim = (1.6, 0.050, 0.750)
        self.table_box_pos = (0, -0.4001, -0.025)
        self.table_box_dim = (1.6, 0.800, 0.750)

        # viz
        self.object_color = [90 / 255, 94 / 255, 173 / 255]
        self.object_color_2 = [150 / 255, 150 / 255, 150 / 255]
        self.indicator_color = [0 / 255, 200 / 255, 200 / 255]

        # mode
        self.enable_evaluation = enable_evaluation
        if self.enable_evaluation:
            self.obj_interior_store = {}
            self.fluid_particle_size = fluid_particle_size
            self.fluid_particle_count = fluid_particle_count
            self.fluid_color = [0.9804, 0.3569, 0.3216]

        # verbose
        self.verbose = verbose

    def _load_obj_model(self, obj_id: str):
        obj_filedir = os.path.join(self.data_filedir, "obj_raw", obj_id)
        obj_filepath = os.path.join(obj_filedir, f"{obj_id}.obj")
        obj_model = trimesh.load(obj_filepath, process=False, force="mesh")
        self.obj_model_store[obj_id] = obj_model
        return obj_model

    def _load_obj_interior(self, obj_id: str):
        obj_filedir = os.path.join(self.data_filedir, "obj_interior", obj_id)
        obj_filepath = os.path.join(obj_filedir, f"{obj_id}.obj")
        obj_model = trimesh.load(obj_filepath, process=False, force="mesh")
        self.obj_interior_store[obj_id] = obj_model
        return obj_model

    def _handle_table(self):
        _tf = np.eye(4, dtype=np.float32)
        _tf[:3, 3] = np.array(
            (float(self.table_box_pos[0]), float(self.table_box_pos[1]), float(self.table_box_pos[2])), dtype=np.float32
        )
        table_def = {
            "id": "table",
            "transf": _tf,
            **load_box_def(box_dim=self.table_box_dim),
            "dof_pos": None,
            "static": True,
            "collision_filter": self.table_collision_filter,
        }
        return table_def

    def _handle_support(self, support_id, static_pos, static_size):
        _tf = np.eye(4, dtype=np.float32)
        _tf[:3, 3] = np.array((float(static_pos[0]), float(static_pos[1]), float(static_pos[2])), dtype=np.float32)
        static_size = (static_size[0] * 2, static_size[1] * 2, static_size[2] * 2)
        support_def = {
            "id": support_id,
            "transf": _tf,
            **load_box_def(box_dim=static_size),
            "dof_pos": None,
            "static": True,
            "collision_filter": self.table_collision_filter,
        }
        return support_def

    def _handle_viz_sphere(self, support_id, tsl, radius):
        _tf = np.eye(4, dtype=np.float32)
        _tf[:3, 3] = np.array((float(tsl[0]), float(tsl[1]), float(tsl[2])), dtype=np.float32)
        viz_sphere_info = {
            "id": support_id,
            "transf": _tf,
            **load_sphere_def(radius=radius),
            "dof_pos": None,
            "static": False,
            "collision_filter": self.table_collision_filter,
            "disable_gravity": True,
            "no_collide": True,
            "color": self.indicator_color,
        }
        return viz_sphere_info

    def _handle_fluid_particle(self, particle_id: str, center_offset: np.ndarray, container_transf: np.ndarray):
        _tf = np.eye(4, dtype=np.float32)
        _tf[:3, 3] = np.array(
            (float(center_offset[0]), float(center_offset[1]), float(center_offset[2])), dtype=np.float32
        )
        particle_tf = container_transf @ _tf
        fluid_sphere_info = {
            "id": f"fluid_{particle_id:0>3}",
            "transf": particle_tf,
            **load_sphere_def(radius=self.fluid_particle_size),
            "dof_pos": None,
            "static": False,
            "collision_filter": self.table_collision_filter,
            "disable_gravity": False,
            "color": self.fluid_color,
        }
        return fluid_sphere_info

    def _handle_fluid(self, container_id: str, container_transf: np.ndarray):
        container_interior = self.obj_interior_store[container_id]
        container_interior_corners = np.asanyarray(container_interior.bounding_box.vertices).copy()
        container_interior_aabb = np.asanyarray(container_interior.bounding_box.bounds)
        container_interior_center = np.mean(container_interior_corners, axis=0)
        container_interior_size = container_interior_aabb[1] - container_interior_aabb[0]
        fluid_block_size = 0.5 * container_interior_size
        fluid_particle_dim = (fluid_block_size / (1.5 * self.fluid_particle_size)).astype(np.int32)
        _grid_dim = int(np.ceil(np.power(self.fluid_particle_count, 1.0 / 3.0)))
        grid_dim = np.array([_grid_dim, _grid_dim, _grid_dim], dtype=np.int32)
        fluid_particle_dim = np.minimum(fluid_particle_dim, grid_dim)
        num_fluid_particle = np.prod(fluid_particle_dim)
        # generate meshgrid inside fluid block: centered at origin, with size of fluid_block_size
        fluid_particle_center = np.stack(
            np.meshgrid(
                np.linspace(-fluid_block_size[0] / 2, fluid_block_size[0] / 2, fluid_particle_dim[0]),
                np.linspace(-fluid_block_size[1] / 2, fluid_block_size[1] / 2, fluid_particle_dim[1]),
                np.linspace(-fluid_block_size[2] / 2, fluid_block_size[2] / 2, fluid_particle_dim[2]),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((num_fluid_particle, 3))
        # check only centers that lies within the interior
        in_interior = check_sign(
            torch.as_tensor(container_interior.vertices, dtype=torch.float32).unsqueeze(0),  # (N, V, 3)
            torch.as_tensor(container_interior.faces, dtype=torch.long),  # (F, 3)
            torch.as_tensor(fluid_particle_center, dtype=torch.float32).unsqueeze(0),  # (N, P, 3)
        )
        fluid_particle_center = fluid_particle_center[in_interior.squeeze(0).numpy()]
        num_fluid_particle = fluid_particle_center.shape[0]
        # get list of fluid particle info
        info_list = []
        for i in range(num_fluid_particle):
            fluid_particle_info = self._handle_fluid_particle(i, fluid_particle_center[i], container_transf)
            info_list.append(fluid_particle_info)
        return info_list

    @staticmethod
    def _to_def(info, extra=None):
        if extra is None:
            extra = {}
        res = {
            "geom_cata": info["geom_cata"],
            "geom_param": info["geom_param"],
            "arti": info["dof_pos"] is not None,
            "static": info["static"],
        }
        for k in ["disable_gravity", "vhacd", "submesh", "collapse_fixed_joints", "no_collide", "color"]:
            if k in info:
                res[k] = info[k]
        res.update(extra)
        return res

    @staticmethod
    def _to_hand_info(raw):
        root_transf = np.asarray(raw["root_transf"])
        tsl = root_transf[:3, 3]
        quat = rotmat_to_quat_np(root_transf[:3, :3])
        qpos_val = raw["qpos"]
        dof_pos = np.concatenate([np.asarray(v) for v in qpos_val.values()], axis=0)
        info = {
            "tsl": tsl,
            "quat": quat,
            "dof_pos": dof_pos,
        }
        return info

    def __call__(self):
        from ..env.embodiment import ShadowHand__NoForearm

        res = {
            "env_info": [],
            "obj_def": {},
            "obj_aux_def": {},
            "lh_def": None,
            "rh_def": {
                "cata": "shadowhand_no_forearm",
                "load_context": ShadowHand__NoForearm,
                "static": False,
                "thickness": 0.001,
                "geom_param": {
                    "asset_dir": os.path.join(_shadowhand_filedir, "hand"),
                    "asset_name": "shadow_hand.xml",
                },
            },
            # payload
            "env_ref_info": [],
        }
        if self.enable_evaluation:
            res["evaluation_info"] = []
        for env_id in wrap_pbar(range(self.num_env), verbose=self.verbose, total=self.num_env, desc="load: "):
            task_offset = env_id % len(self.task_info_list)

            task_hashcode = hash_json_attr_map(self.task_info_list[task_offset])
            if task_hashcode in self.cache_dict:
                res["env_info"].append(self.cache_dict[task_hashcode]["env_info"])
                res["obj_def"].update(
                    {
                        obj_id: def_item
                        for obj_id, def_item in self.cache_dict[task_hashcode]["obj_def"].items()
                        if obj_id not in res["obj_def"]
                    }
                )
                res["obj_aux_def"].update(
                    {
                        obj_aux_id: aux_def_item
                        for obj_aux_id, aux_def_item in self.cache_dict[task_hashcode]["obj_aux_def"].items()
                        if obj_aux_id not in res["obj_aux_def"]
                    }
                )
                res["env_ref_info"].append(self.cache_dict[task_hashcode]["env_ref_info"])
                if self.enable_evaluation:
                    res["evaluation_info"].append(self.cache_dict[task_hashcode]["evaluation_info"])
                continue
            self.cache_dict[task_hashcode] = {}

            # region: load
            task_def = self.task_info_list[task_offset]
            task_primitive = task_def["primitive"]
            task_code = task_def["code"]
            process_key, task_type, task_range, task_main_object = parse_task_code(task_code)
            with open(os.path.join(self.data_filedir, task_primitive, "meta", f"{task_code}.json"), "r") as ifs:
                task_meta = json.load(ifs)
            with open(
                os.path.join(self.data_filedir, task_primitive, "ref_raw", task_type, f"{task_code}.pkl"), "rb"
            ) as ifs:
                task_ref_raw = pickle.load(ifs)

            # endregion: load

            # fill in item
            item = {
                "process_key": process_key,
                "frame_id": task_range[0],
                "obj_info": [],
                "obj_info_aux": [],
                "lh_info": None,
                "rh_info": None,
                "task_info": None,
            }
            if self.enable_evaluation:
                eval_item = {}

            # region: env_info
            # fill the table
            cached_obj_aux_def_item, cached_obj_def_item = {}, {}
            table_info = self._handle_table()
            if table_info is not None:
                item["obj_info_aux"].append(table_info)
                cached_obj_aux_def_item["table"] = self._to_def(table_info)

            if len(task_ref_raw["obj_aux_def_info"]) > 0:
                for obj_aux_id, obj_aux_def_info in task_ref_raw["obj_aux_def_info"].items():
                    obj_aux_id_full = f"{task_code}++{obj_aux_id}"
                    support_info = self._handle_support(
                        obj_aux_id_full, obj_aux_def_info["static_pos"], obj_aux_def_info["static_size"]
                    )
                    item["obj_info_aux"].append(support_info)
                    cached_obj_aux_def_item[obj_aux_id_full] = self._to_def(support_info)

            obj_id = task_main_object
            obj_transf_init = task_ref_raw["obj_def_info"][obj_id]["init"]
            obj_info = {
                "id": obj_id,
                "transf": obj_transf_init,
                "geom_cata": GeomCata.URDF,
                "geom_param": {
                    "asset_dir": os.path.normpath(os.path.join(_data_filedir, "obj", obj_id)),
                    "asset_name": f"coacd/coacd.urdf",
                },
                "dof_pos": None,
                "static": False,
                "collision_filter": self.obj_collision_filter,
                "vhacd": False,
                "submesh": True,
                "collapse_fixed_joints": True,
            }
            item["obj_info"].append(obj_info)
            cached_obj_def_item[obj_id] = self._to_def(obj_info, extra={"color": self.object_color})
            self._load_obj_model(obj_id)

            obj_src_id = task_meta["tool_src"]
            if obj_src_id != obj_id:
                raise NotImplementedError()
            if self.enable_evaluation:
                # load interior of obj_src
                self._load_obj_interior(obj_id)
                # generate aux spheres for fluid
                _fluid_info_list = self._handle_fluid(obj_id, obj_transf_init)
                item["obj_info_aux"].extend(_fluid_info_list)
                for _info_item in _fluid_info_list:
                    cached_obj_aux_def_item[_info_item["id"]] = self._to_def(_info_item)
                eval_item["num_fluid_particle"] = len(_fluid_info_list)

            obj_dst_id = task_meta["tool_dst"]
            if self.enable_evaluation:
                obj_dst_transf_init = task_ref_raw["obj_def_info"][obj_dst_id]["target"]
                obj_dst_info = {
                    "id": obj_dst_id,
                    "transf": obj_dst_transf_init,
                    "geom_cata": GeomCata.URDF,
                    "geom_param": {
                        "asset_dir": os.path.normpath(os.path.join(_data_filedir, "obj", obj_dst_id)),
                        "asset_name": f"coacd/coacd.urdf",
                    },
                    "dof_pos": None,
                    "static": True,
                    "collision_filter": self.obj_collision_filter,
                    "vhacd": False,
                    "submesh": True,
                    "collapse_fixed_joints": True,
                }
                item["obj_info"].append(obj_dst_info)  # !! REDUCE NOISE
                cached_obj_def_item[obj_dst_id] = self._to_def(obj_dst_info, extra={"color": self.object_color_2})
                # load interior of obj_dst
                self._load_obj_interior(obj_dst_id)

            root_transf = np.asarray(task_ref_raw["hand_def_info"]["rh"]["init"]["root_transf"])
            tsl = root_transf[:3, 3]
            quat = rotmat_to_quat_np(root_transf[:3, :3])
            qpos_val = task_ref_raw["hand_def_info"]["rh"]["init"]["qpos"]
            dof_pos = np.concatenate([np.asarray(v) for v in qpos_val.values()], axis=0)
            item["rh_info"] = {
                "static": False,
                "tsl": tsl,
                "quat": quat,
                "dof_pos": dof_pos,
                "collision_filter": self.hand_collision_filter,
            }

            # task_info
            item["task_info"] = {
                "obj_id": obj_id,
                "obj_src_id": obj_src_id,
                "obj_dst_id": obj_dst_id,
            }

            # store
            res["env_info"].append(item)
            self.cache_dict[task_hashcode]["env_info"] = item
            self.cache_dict[task_hashcode]["obj_aux_def"] = cached_obj_aux_def_item
            self.cache_dict[task_hashcode]["obj_def"] = cached_obj_def_item
            if self.enable_evaluation:
                res["evaluation_info"].append(eval_item)
                self.cache_dict[task_hashcode]["evaluation_info"] = eval_item
            # endregion: env_info

            # region: env_ref_info
            # handle payload
            ref_item = {}
            res["env_ref_info"].append(ref_item)
            self.cache_dict[task_hashcode]["env_ref_info"] = ref_item
            # endregion: env_ref_info

            # region: collect def
            res["obj_def"].update(
                {obj_id: def_item for obj_id, def_item in cached_obj_def_item.items() if obj_id not in res["obj_def"]}
            )
            res["obj_aux_def"].update(
                {
                    obj_aux_id: aux_def_item
                    for obj_aux_id, aux_def_item in cached_obj_aux_def_item.items()
                    if obj_aux_id not in res["obj_aux_def"]
                }
            )
            res["obj_model"] = self.obj_model_store.copy()
            if self.enable_evaluation:
                res["obj_interior"] = self.obj_interior_store.copy()
            # endregion: collect def

        return res
