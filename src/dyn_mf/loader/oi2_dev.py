from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing import Optional, Mapping, Any

import os
import numpy as np
import torch
import pickle
import json

import pytorch_kinematics
from dev_fn.transform.rotation_np import rotmat_to_quat_np
from dev_fn.transform.transform_np import assemble_T_np
from ..scene import FrameConvention, GeomCata
from .common import load_table_def, load_box_def
from .common import AssetLoader

_thisdir = os.path.dirname(__file__)
_shadowhand_filedir = os.path.normpath(os.path.join(_thisdir, "../../..", "asset/shadowhand_no_forearm"))
_data_filedir = os.path.normpath(os.path.join(_thisdir, "../../..", "asset/oi2_dev"))


def parse_task_code(task_code):
    task_code_list = task_code.split("++")
    process_key = f"{task_code_list[0]}/{task_code_list[1]}"
    task_type = task_code_list[2]
    task_range_str_list = task_code_list[3].split("_")
    task_range = (int(task_range_str_list[0]), int(task_range_str_list[1]))
    task_main_object = task_code_list[4]
    return process_key, task_type, task_range, task_main_object


class OakInk2DevAssetLoader(AssetLoader):
    def __init__(
        self,
        num_env: int,
        task_info_list: list[Mapping[str, Any]],
        data_filedir: str = _data_filedir,
        shadowhand_filedir: str = _shadowhand_filedir,
    ) -> None:
        self.num_env = num_env
        self.data_filedir = data_filedir
        self.shadowhand_filedir = shadowhand_filedir
        self.task_info_list = task_info_list
        self.frame_convention = (FrameConvention.Y_UP,)

        self.hand_type = "rh"

        self.hand_collision_filter = 0b0000
        self.obj_collision_filter = 0b0000
        self.table_collision_filter = 0b0000

        # table
        self.table_box_pos = (0, -0.02501, -0.025)
        self.table_box_dim = (1.6, 0.050, 0.750)

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

    @staticmethod
    def _to_def(info):
        res = {
            "geom_cata": info["geom_cata"],
            "geom_param": info["geom_param"],
            "arti": info["dof_pos"] is not None,
            "static": info["static"],
        }
        for k in ["vhacd", "submesh", "collapse_fixed_joints"]:
            if k in info:
                res[k] = info[k]
        return res

    def __call__(self):
        res = {
            "env_info": [],
            "obj_def": {},
            "obj_aux_def": {},
            "lh_def": None,
            "rh_def": {
                "cata": "shadowhand_no_forearm",
                "static": False,
                "thickness": 0.001,
                "geom_param": {
                    "asset_dir": os.path.join(_shadowhand_filedir, "hand"),
                    "asset_name": "shadow_hand.xml",
                },
            },
        }
        for env_id in range(self.num_env):
            task_offset = env_id % len(self.task_info_list)

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

            item = {
                "process_key": process_key,
                "frame_id": task_range[0],
                "obj_info": [],
                "obj_info_aux": [],
                "lh_info": None,
                "rh_info": None,
            }

            # fill the table
            table_def = self._handle_table()
            if table_def is not None:
                item["obj_info_aux"].append(table_def)
                if "table" not in res["obj_aux_def"]:
                    res["obj_aux_def"]["table"] = self._to_def(table_def)

            if len(task_ref_raw["obj_aux_def_info"]) > 0:
                for obj_aux_id, obj_aux_def_info in task_ref_raw["obj_aux_def_info"].items():
                    obj_aux_id_full = f"{task_code}++{obj_aux_id}"
                    support_def = self._handle_support(
                        obj_aux_id_full, obj_aux_def_info["static_pos"], obj_aux_def_info["static_size"]
                    )
                    item["obj_info_aux"].append(support_def)
                    if obj_aux_id_full not in res["obj_aux_def"]:
                        res["obj_aux_def"][obj_aux_id_full] = self._to_def(support_def)

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
            if obj_id not in res["obj_def"]:
                res["obj_def"][obj_id] = self._to_def(obj_info)

            obj_dst_id = task_meta["tool_dst"]
            obj_dst_transf_init = task_ref_raw["obj_def_info"][obj_dst_id]["init"]
            obj_dst_info = {
                "id": obj_dst_id,
                "transf": obj_dst_transf_init,
                "geom_cata": GeomCata.URDF,
                "geom_param": {
                    "asset_dir": os.path.normpath(os.path.join(_data_filedir, "obj", obj_dst_id)),
                    "asset_name": f"coacd/coacd.urdf",
                },
                "dof_pos": None,
                "static": False,
                "collision_filter": self.obj_collision_filter,
                "vhacd": False,
                "submesh": True,
                "collapse_fixed_joints": True,
            }
            item["obj_info"].append(obj_dst_info)
            if obj_dst_id not in res["obj_def"]:
                res["obj_def"][obj_dst_id] = self._to_def(obj_dst_info)

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

            # handle obj_aux
            res["env_info"].append(item)
        return res
