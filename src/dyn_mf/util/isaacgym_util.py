from __future__ import annotations

import os
import sys
from dev_fn.util.console_io import RedirectStream

with RedirectStream(), RedirectStream(sys.stderr):
    from isaacgym import gymapi, gymutil, gymtorch
import torch

from dev_fn.transform.rotation_jit import quat_apply


def debug_viz_frame(gym, viewer, env_ptr, tsl: torch.Tensor, quat: torch.Tensor):
    device = tsl.device
    dtype = tsl.dtype

    posx = (tsl + quat_apply(quat, torch.as_tensor([1, 0, 0], device=device, dtype=dtype) * 0.2)).cpu().numpy()
    posy = (tsl + quat_apply(quat, torch.as_tensor([0, 1, 0], device=device, dtype=dtype) * 0.2)).cpu().numpy()
    posz = (tsl + quat_apply(quat, torch.as_tensor([0, 0, 1], device=device, dtype=dtype) * 0.2)).cpu().numpy()

    p0 = tsl.cpu().numpy()
    gym.add_lines(viewer, env_ptr, 2, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
    gym.add_lines(viewer, env_ptr, 2, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
    gym.add_lines(viewer, env_ptr, 2, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])


def contact_get_env(contact_info):
    env_0 = contact_info['env0']
    env_1 = contact_info['env1']
    if env_0 != -1 and env_1 != -1 and env_0 == env_1:
        return env_0
    else:
        return None
