from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from .general import EnvGeneral

import numpy as np
import torch


class EnvCallbackContext(object):
    @property
    def obs_dim(self):
        pass

    @property
    def act_dim(self):
        pass

    @property
    def timeout(self):
        pass

    def attach_env(self, env: EnvGeneral):
        self.env = env

    def detach_env(self):
        self.env = None

    # reset_idx
    def reset_idx(self, env_id_to_reset: torch.Tensor):
        pass

    # action
    def apply_action(self, action: torch.Tensor, env_id_to_reset: torch.Tensor):
        pass

    # reward
    def compute_reward(self, action: torch.Tensor):
        pass

    # obs
    def compute_observation(self):
        pass

    # debug viz
    def viz_debug(self):
        pass
