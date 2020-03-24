from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DotmapUtils import get_required_argument

from config.ensemble_model import EnsembleModel
import gym
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DuckietownConfigModule:
    ENV_NAME = "MBRLDuckietown-v0"
    TASK_HORIZON = 125
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    NTEST_ROLLOUTS = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 6, 4
    GP_NINDUCING_POINTS = 200
    ROAD_TILE_SIZE = 0.585
    GOAL_REWARD = 100
    CATASTROPHE_COST = 10000 #This is used in MPC planning for avoiding collisions
    COLLISION_SIGMOID = torch.nn.Sigmoid()
    MODEL_ENSEMBLE_SIZE = 4
    MODEL_HIDDEN_SIZE = 200
    MODEL_WEIGHT_DECAYS = [1e-4, 2.5e-4, 2.5e-4, 5e-4]


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
        # ... catastrophe, goal_x, goal_y
        return obs[..., :-3]

    @staticmethod
    def obs_postproc(obs, pred):
        ret = obs[..., :-3] + pred[..., :-1]
        ret = torch.cat((ret, CONFIG_MODULE.COLLISION_SIGMOID(pred[..., -1:]), obs[..., -2:]), dim=-1)
        # this gives us output prediction, catastrophe probability, goal_x, goal_y
        return ret

    @staticmethod
    def targ_proc(obs, next_obs):   # This is to undo obs_postproc
        return np.concatenate(((next_obs - obs)[..., :-3], next_obs[..., -3:-2]), axis=-1)

    @staticmethod
    def obs_cost_fn(obs):
        cur_pos, goal = obs[:, 0:2], obs[:, -2:]
        cur_pos_tile = CONFIG_MODULE._get_grid_coords(cur_pos)
        reached_goal_mask = torch.all(cur_pos_tile == goal, dim=-1)
        distances = CONFIG_MODULE._get_manhattan_dist_to_goal(cur_pos, goal)
        distances[reached_goal_mask] = -CONFIG_MODULE.GOAL_REWARD
        return distances

    @staticmethod
    def catastrophe_cost_fn(obs, cost, percentile):
        hit_wall_mask =  obs[..., -3] > percentile / 100
        cost[hit_wall_mask] += CONFIG_MODULE.CATASTROPHE_COST
        return cost

    @staticmethod
    def _get_grid_coords(abs_pos):
        grid_pos = torch.floor(abs_pos / CONFIG_MODULE.ROAD_TILE_SIZE)
        return grid_pos

    @staticmethod
    def ac_cost_fn(acs):
        return 0

    @staticmethod
    def _get_manhattan_dist_to_goal(cur_pos, goal):
        """
        Returns minimium manhattan distance to closest point on goal tile
        """
        goal_x_range = torch.cat([goal[:, :1] * CONFIG_MODULE.ROAD_TILE_SIZE,
                                  (goal[:, :1] + 1) * CONFIG_MODULE.ROAD_TILE_SIZE], dim=-1)
        goal_z_range = torch.cat([goal[:, 1:] * CONFIG_MODULE.ROAD_TILE_SIZE,
                                  (goal[:, 1:] + 1) * CONFIG_MODULE.ROAD_TILE_SIZE], dim=-1)
        mask_x = (cur_pos[:, 0] >= goal_x_range[:, 0]) & (cur_pos[:, 0] < goal_x_range[:, 1])
        mask_z = (cur_pos[:, 1] >= goal_z_range[:, 0]) & (cur_pos[:, 1] < goal_z_range[:, 1])
        x_dist = torch.zeros(mask_x.shape).to(TORCH_DEVICE)
        z_dist = torch.zeros(mask_z.shape).to(TORCH_DEVICE)
        x_dist = torch.where(mask_x, x_dist,
                             torch.min(torch.abs(cur_pos[:, :1] - goal_x_range), dim=-1)[0])
        z_dist = torch.where(mask_z, z_dist,
                             torch.min(torch.abs(cur_pos[:, 1:] - goal_z_range), dim=-1)[0])
        return x_dist + z_dist

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


CONFIG_MODULE = DuckietownConfigModule
