from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DotmapUtils import get_required_argument
from config.ensemble_model import EnsembleModel

import gym
import numpy as np
import torch

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CartPoleConfigModule:
    ENV_NAME = "MBRLCartPole-v0"
    TASK_HORIZON = 200
    NTRAIN_ITERS = 50
    NROLLOUTS_PER_ITER = 1
    NTEST_ROLLOUTS = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 6, 4
    GP_NINDUCING_POINTS = 200
    CATASTROPHE_SIGMOID = torch.nn.Sigmoid()
    CATASTROPHE_COST = 10000
    MODEL_ENSEMBLE_SIZE = 5
    MODEL_HIDDEN_SIZE = 500
    MODEL_WEIGHT_DECAYS = [1e-4, 2.5e-4, 2.5e-4, 5e-4]

    # Create and move this tensor to GPU so that
    # we do not waste time moving it repeatedly to GPU later
    ee_sub = torch.tensor([0.0, 0.6], device=TORCH_DEVICE, dtype=torch.float)

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        #.... pendulum_length, catastrophe
        if isinstance(obs, np.ndarray):
            return np.concatenate([np.sin(obs[:, 1:2]), np.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:-2]], axis=1)
        else:
            return torch.cat([torch.sin(obs[:, 1:2]), torch.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:-2]], dim=1)

    @staticmethod
    def obs_postproc(obs, pred):
        return torch.cat((obs[..., :-2] + pred[..., :-1], obs[..., -2:-1],
                          CONFIG_MODULE.CATASTROPHE_SIGMOID(pred[..., -1:])), dim=-1)

    @staticmethod
    def targ_proc(obs, next_obs):   # This is to undo obs_postproc
        if isinstance(obs, np.ndarray):
            return np.concatenate([next_obs[..., :-2] - obs[..., :-2], next_obs[..., -1:]], axis=-1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([next_obs[..., :-2] - obs[..., :-2], next_obs[..., -1:]], dim=-1)

    @staticmethod
    def obs_cost_fn(obs):
        ee_pos, ideal_pos = CONFIG_MODULE._get_ee_pos(obs)

        ee_pos -= ideal_pos

        ee_pos = ee_pos ** 2

        ee_pos = - ee_pos.sum(dim=1)

        pendulum_length = obs[:, -2:-1].squeeze(-1)

        return - (ee_pos / (pendulum_length ** 2)).exp()

    @staticmethod
    def ac_cost_fn(acs):
        return 0.01 * (acs ** 2).sum(dim=1)

    @staticmethod
    def _get_ee_pos(obs):
        x0, theta, pendulum_length = obs[:, :1], obs[:, 1:2], obs[:, -2:-1]
        ee_pos = torch.cat([x0 + pendulum_length * torch.sin(theta), pendulum_length * torch.cos(theta)], dim=1)
        ideal_pos = torch.cat((torch.zeros_like(pendulum_length), pendulum_length), dim=-1)
        return ee_pos, ideal_pos

    def nn_constructor(self, model_init_cfg):

        ensemble_size = get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size")

        load_model = model_init_cfg.get("load_model", False)

        assert load_model is False, 'Has yet to support loading model'

        model = EnsembleModel(ensemble_size,
                        in_features=self.MODEL_IN,
                        out_features=self.MODEL_OUT * 2 + 1, 
                        hidden_size=self.MODEL_HIDDEN_SIZE,
                        num_layers=len(self.MODEL_WEIGHT_DECAYS),
                        weight_decays=self.MODEL_WEIGHT_DECAYS).to(TORCH_DEVICE)

        model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

        return model

    @staticmethod
    def catastrophe_cost_fn(obs, cost, percentile):
        catastrophe_mask = obs[..., -1] > percentile / 100
        cost[catastrophe_mask] += CONFIG_MODULE.CATASTROPHE_COST
        return cost

CONFIG_MODULE = CartPoleConfigModule
