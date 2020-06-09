from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DotmapUtils import get_required_argument

import gym
import numpy as np
from config.ensemble_model import EnsembleModel
import torch
from torch import nn as nn

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class HalfCheetahConfigModule:
    ENV_NAME = "MBRLHalfCheetahDisabled-v0"
    TASK_HORIZON = 1000
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    NTEST_ROLLOUTS = 1
    PLAN_HOR = 10
    MODEL_IN, MODEL_OUT = 25, 20
    COLLISION_COST = 10000
    MODEL_ENSEMBLE_SIZE = 5
    MODEL_HIDDEN_SIZE = 200
    COLLISION_SIGMOID = torch.nn.Sigmoid()
    MODEL_WEIGHT_DECAYS = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.NN_TRAIN_CFG = {"epochs": 10}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }

}

    @staticmethod
    def obs_preproc(obs):
        #... reward, catastrophe_predictor
        return obs[..., :-2]

    @staticmethod
    def obs_postproc(obs, pred):

        assert isinstance(obs, torch.Tensor)

        return torch.cat([
            obs[:, :-2] + pred[:, :-2],
            pred[:, -2:-1],
            CONFIG_MODULE.COLLISION_SIGMOID(pred[:, -1:]),
        ], dim=1)

    @staticmethod
    def targ_proc(obs, next_obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([next_obs[:, :-2] - obs[:, :-2], next_obs[:, -2:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                next_obs[:, :-2] - obs[:, :-2],
                next_obs[:, -2:],
            ], dim=1)

    @staticmethod
    def obs_cost_fn(obs):
        return -obs[:, -2]

    @staticmethod
    def catastrophe_cost_fn(obs, cost, percentile):
        catastrophe_mask = obs[..., -1] > percentile / 100
        cost[catastrophe_mask] += CONFIG_MODULE.COLLISION_COST
        return cost

    @staticmethod
    def ac_cost_fn(acs):
        return 1e-1 * 0.5 * (acs ** 2).sum(dim=1)

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


CONFIG_MODULE = HalfCheetahConfigModule
