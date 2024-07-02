from __future__ import annotations

import sys
import os
import math
import numpy as np
import pickle
import random
import json

import logging
from dev_fn.upkeep import log
from dev_fn.util.console_io import RedirectStream
from dev_fn.util.console_io import suppress_gym_logging, suppress_trimesh_logging, filter_warnings

with RedirectStream(), RedirectStream(sys.stderr):
    from isaacgym import gymapi, gymutil, gymtorch
import torch

from dev_fn.util import xml_util
import xml
import xml.etree.ElementTree as ET

from dev_fn.dataset.process_upkeep import ProcessDef
from dev_fn.transform.rotation_np import rotmat_to_quat_np

from dyn_mf.env.general import EnvGeneral
from dyn_mf.loader.oi2_dev import OakInk2DevAssetLoader
from dyn_mf.env.oi2_dev import OakInk2DevEnvCallbackContext
from manotorch.manolayer import ManoLayer

_logger = logging.getLogger(__name__)


def main():
    log.log_init()
    log.enable_console()

    suppress_trimesh_logging()
    suppress_gym_logging()
    filter_warnings(cata=DeprecationWarning, module="mano")
    filter_warnings(cata=DeprecationWarning, module="chumpy")

    device = "cuda:0"
    num_envs = 100
    random.seed(0)

    with open("asset/oi2_dev/pour/task_list_rh__train.json", "r") as ifs:
        code_list = json.load(ifs)
    task_info_list = []
    for code in code_list:
        task_info_list.append(
            {
                "primitive": "pour",
                "code": code,
            }
        )
    asset_loader = OakInk2DevAssetLoader(num_env=num_envs, task_info_list=task_info_list)
    env_callback_ctx = OakInk2DevEnvCallbackContext(num_env=num_envs, device=device, dtype=torch.float32)

    env_cfg = dict(
        isaacgym_cfg=dict(
            env=dict(
                numEnvs=num_envs,
                enableDebugVis=True,
            ),
        ),
    )
    env = EnvGeneral(
        isaacgym_cfg=env_cfg["isaacgym_cfg"],
        rl_device=device,
        sim_device=device,
        graphics_device_id=0,  # on platform10 -> cuda:3 is graphics_device 0
        headless=False,
        force_render=True,
        asset_loader=asset_loader,
        env_callback_ctx=env_callback_ctx,
    )

    obs_dict = env.reset()
    while True:
        obs = obs_dict["obs"]
        action = torch.rand((num_envs,) + env.action_space.shape, device=device)
        # action = action * 2 - 1.0
        action[:, :] = 0.0
        obs_dict, rew, done, info = env.step(action)

    log.log_deinit()


if __name__ == "__main__":
    main()
