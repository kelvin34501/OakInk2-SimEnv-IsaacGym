from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing import Optional, Any, Union
    from torch import Tensor
    from numpy import ndarray

import os
import numpy as np
import torch
import dataclasses

from .process_upkeep import ProcessDef
from .segment import HAND_SIDE
from manotorch.manolayer import ManoLayer, MANOOutput
from dev_fn.util.dataclass_util import NamedData


@dataclasses.dataclass
class FrameMano(NamedData):
    pose: Tensor
    tsl: Tensor
    shape: Tensor


@dataclasses.dataclass
class FrameManoBH(NamedData):
    rh: FrameMano
    lh: FrameMano


@dataclasses.dataclass
class FrameManoOut(NamedData):
    j: Tensor
    v: Tensor


@dataclasses.dataclass
class FrameManoOutBH(NamedData):
    rh: FrameManoOut
    lh: FrameManoOut


def load_frame_res_from_path(save_filepath: str, dtype: torch.dtype, device: torch.device):
    if os.path.exists(save_filepath):
        frame_res = torch.load(save_filepath, map_location=device)
        for _cata, _res in frame_res.items():
            if isinstance(_res, dict):
                for _k in _res:
                    if torch.is_tensor(_res[_k]):
                        _res[_k] = _res[_k].to(dtype)
            elif torch.is_tensor(_res):
                frame_res[_cata] = _res.to(dtype)
        return frame_res
    else:
        return None


def load_frame_res(save_prefix: str, process_def: ProcessDef, frame_id: int, dtype: torch.dtype, device: torch.device):
    process_def_offset = process_def.to_offset()
    load_filepath = os.path.join(save_prefix, process_def_offset, f"{frame_id:0>6}.pt")
    return load_frame_res_from_path(load_filepath, dtype, device)


class StreamDatasetMano(torch.utils.data.Dataset):
    def __init__(self, process_key: str, frame_id_list: list[int], mano_res_path: str):
        self.process_key = process_key
        self.process_def = ProcessDef(self.process_key)
        self.frame_id_list = frame_id_list
        self.mano_res_prefix = mano_res_path

        self.store = {}
        for hand_side in HAND_SIDE:
            seg_mano_pose_traj = []
            seg_mano_tsl_traj = []
            seg_mano_shape_traj = []
            for frame_id in self.frame_id_list:
                _frame_res = load_frame_res(
                    self.mano_res_prefix, self.process_def, frame_id, torch.float32, torch.device("cpu")
                )
                _frame_input = _frame_res["input"]
                _pose = _frame_input[f"{hand_side}__pose_coeffs"]
                _tsl = _frame_input[f"{hand_side}__tsl"]
                _shape = _frame_input[f"{hand_side}__betas"]
                seg_mano_pose_traj.append(_pose)
                seg_mano_tsl_traj.append(_tsl)
                seg_mano_shape_traj.append(_shape)

            self.store[hand_side] = {
                "pose": seg_mano_pose_traj,
                "tsl": seg_mano_tsl_traj,
                "shape": seg_mano_shape_traj,
            }

    def __getitem__(self, index):
        res = {}
        for hand_side in HAND_SIDE:
            res[hand_side] = FrameMano(
                **{
                    "pose": self.store[hand_side]["pose"][index],
                    "tsl": self.store[hand_side]["tsl"][index],
                    "shape": self.store[hand_side]["shape"][index],
                }
            )
        res = FrameManoBH(**res)
        return res

    def __len__(self):
        return len(self.frame_id_list)


def batch_tensor_generator(tensor, batch_size):
    num_full_batch = tensor.size(0) // batch_size
    for i in range(num_full_batch):
        yield tensor[i * batch_size : (i + 1) * batch_size]
    if tensor.size(0) % batch_size != 0:
        yield tensor[num_full_batch * batch_size :]


class StreamDatasetManoOut(torch.utils.data.Dataset):
    def __init__(self, mano_layer_rh, mano_layer_lh, device, dtype, mano_data, runtime_batch_size=100):
        self.len = len(mano_data)
        # for each hand, concat pose tsl shape
        # slice into list of blocks and pass given mano_layer
        # cache the verts and faces
        traj_j_store, traj_v_store = {}, {}
        mano_layer_store = {
            "rh": mano_layer_rh,
            "lh": mano_layer_lh,
        }
        for hand_side in HAND_SIDE:
            pose_th = torch.cat(mano_data.store[hand_side]["pose"], dim=0)
            tsl_th = torch.cat(mano_data.store[hand_side]["tsl"], dim=0)
            shape_th = torch.cat(mano_data.store[hand_side]["shape"], dim=0)
            traj_len = pose_th.shape[0]

            j_list, v_list = [], []
            for pose_sl, tsl_sl, shape_sl in zip(
                batch_tensor_generator(pose_th, runtime_batch_size),
                batch_tensor_generator(tsl_th, runtime_batch_size),
                batch_tensor_generator(shape_th, runtime_batch_size),
            ):
                pose_sl = pose_sl.to(device=device, dtype=dtype)
                tsl_sl = tsl_sl.to(device=device, dtype=dtype)
                shape_sl = shape_sl.to(device=device, dtype=dtype)
                with torch.no_grad():
                    mano_out_sl = mano_layer_store[hand_side](pose_coeffs=pose_sl, betas=shape_sl)
                    j_sl = mano_out_sl.joints + tsl_sl.unsqueeze(1)
                    v_sl = mano_out_sl.verts + tsl_sl.unsqueeze(1)
                j = j_sl.clone().cpu().numpy()
                v = v_sl.clone().cpu().numpy()
                j_list.append(j)
                v_list.append(v)
            j_list = np.concatenate(j_list, axis=0)
            v_list = np.concatenate(v_list, axis=0)
            traj_j_store[hand_side] = j_list
            traj_v_store[hand_side] = v_list
        self.traj_j_store, self.traj_v_store = traj_j_store, traj_v_store

    def __getitem__(self, index):
        res = {}
        for hand_side in HAND_SIDE:
            j = self.traj_j_store[hand_side][index]
            v = self.traj_v_store[hand_side][index]
            res[hand_side] = FrameManoOut(**{"j": j, "v": v})
        res = FrameManoOutBH(**res)
        return res

    def __len__(self):
        return self.len
