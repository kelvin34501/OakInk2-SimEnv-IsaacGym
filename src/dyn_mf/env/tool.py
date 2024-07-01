import numpy as np
import torch

from dev_fn.transform.rotation_np import quat_to_euler_angle_np
from dev_fn.transform.rotation import quat_to_rotmat
from dev_fn.transform.transform import assemble_T


def assemble_full_dof_pos(info_item, dtype, device, zero_root=False):
    if not zero_root:
        tsl = torch.as_tensor(info_item["tsl"].reshape((3,)), dtype=dtype, device=device)
        _euler = quat_to_euler_angle_np(info_item["quat"], "XYZ")
        euler = torch.as_tensor(_euler.reshape((3,)), dtype=dtype, device=device)
    else:
        tsl = torch.zeros((3,), dtype=dtype, device=device)
        euler = torch.zeros((3,), dtype=dtype, device=device)
    dof_pos = torch.as_tensor(
        info_item["dof_pos"].reshape((-1,)),
        dtype=dtype,
        device=device,
    )
    full_dof_pos = torch.cat((tsl, euler, dof_pos), dim=0)
    return full_dof_pos


try:
    import pytorch_kinematics
except ImportError:
    pass


def forward_kinematic_chain(
    chain: pytorch_kinematics.Chain,
    *,
    transf: torch.Tensor = None,
    dof_pos: torch.Tensor,
    tsl: torch.Tensor = None,
    quat: torch.Tensor = None,
):
    if transf is None:
        rotmat = quat_to_rotmat(quat)
        transf = assemble_T(tsl, rotmat)
    ret = chain.forward_kinematics(dof_pos)

    res = {}
    for frame_name, frame_transform3d in ret.items():
        local_frame_transf = frame_transform3d.get_matrix()
        frame_transf = transf @ local_frame_transf
        res[frame_name] = frame_transf

    return res


def compute_contact_wordaround(force_tensor, thres=1e-3):
    contact = (force_tensor.norm(dim=-1) > thres).to(force_tensor.dtype)
    return contact
