from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.io import savemat

from DotmapUtils import get_required_argument
from optimizers import CEMOptimizer, DiscreteCEMOptimizer, DiscreteRandomOptimizer

import matplotlib.pyplot as plt

from tqdm import trange
from Controller import Controller

import torch

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


class MPC(Controller):

    def __init__(self, params):
        """Creates class instance.

        Arguments:
            params
                .env (gym.env): Environment for which this controller will be used.
                .ac_ub (np.ndarray): (optional) An array of action upper bounds.
                    Defaults to environment action upper bounds.
                .ac_lb (np.ndarray): (optional) An array of action lower bounds.
                    Defaults to environment action lower bounds.
                .per (int): (optional) Determines how often the action sequence will be optimized.
                    Defaults to 1 (reoptimizes at every call to act()).
                .prop_cfg
                    .model_init_cfg (DotMap): A DotMap of initialization parameters for the model.
                        .model_constructor (func): A function which constructs an instance of this
                            model, given model_init_cfg.
                    .model_train_cfg (dict): (optional) A DotMap of training parameters that will be passed
                        into the model every time is is trained. Defaults to an empty dict.
                    .model_pretrained (bool): (optional) If True, assumes that the model
                        has been trained upon construction.
                    .mode (str): Propagation method. Choose between [E, DS, TSinf, TS1, MM].
                        See https://arxiv.org/abs/1805.12114 for details.
                    .npart (int): Number of particles used for DS, TSinf, TS1, and MM propagation methods.
                    .ign_var (bool): (optional) Determines whether or not variance output of the model
                        will be ignored. Defaults to False unless deterministic propagation is being used.
                    .obs_preproc (func): (optional) A function which modifies observations (in a 2D matrix)
                        before they are passed into the model. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and PyTorch arrays.
                    .obs_postproc (func): (optional) A function which returns vectors calculated from
                        the previous observations and model predictions, which will then be passed into
                        the provided cost function on observations. Defaults to lambda obs, model_out: model_out.
                        Note: Must be able to process both NumPy and PyTorch arrays.
                    .obs_postproc2 (func): (optional) A function which takes the vectors returned by
                        obs_postproc and (possibly) modifies it into the predicted observations for the
                        next time step. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .targ_proc (func): (optional) A function which takes current observations and next
                        observations and returns the array of targets (so that the model learns the mapping
                        obs -> targ_proc(obs, next_obs)). Defaults to lambda obs, next_obs: next_obs.
                        Note: Only needs to process NumPy arrays.
                    .continue_train (bool): (optional) Whether or not to continue 
                .opt_cfg
                    .mode (str): Internal optimizer that will be used. Choose between [CEM].
                    .cfg (DotMap): A map of optimizer initializer parameters.
                    .plan_hor (int): The planning horizon that will be used in optimization.
                    .obs_cost_fn (func): A function which computes the cost of every observation
                        in a 2D matrix.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .ac_cost_fn (func): A function which computes the cost of every action
                        in a 2D matrix.
                    .catastrophe_cost_fn (func) A function that computes the cost of catastrophe.
                    .no_catastrophe_pred (bool): Whether or not to train/use catastrophe prediction.
                    .percentile (float): The percentile used for either catastrophic state or reward-based
                        risk aversion.
        """
        super().__init__(params)
        self.dO, self.dU = params.env.observation_space.shape[0], params.env.action_space.shape[0]
        self.ac_ub, self.ac_lb = params.env.action_space.high, params.env.action_space.low
        self.ac_ub = np.minimum(self.ac_ub, params.get("ac_ub", self.ac_ub))
        self.ac_lb = np.maximum(self.ac_lb, params.get("ac_lb", self.ac_lb))
        self.update_fns = params.get("update_fns", [])
        self.per = params.get("per", 1)

        self.model_init_cig = params.prop_cfg.get("model_init_cfg", {})
        self.model_train_cfg = params.prop_cfg.get("model_train_cfg", {})
        self.prop_mode = get_required_argument(params.prop_cfg, "mode", "Must provide propagation method.")
        self.npart = get_required_argument(params.prop_cfg, "npart", "Must provide number of particles.")
        self.ign_var = params.prop_cfg.get("ign_var", False) or self.prop_mode == "E"

        self.obs_preproc = params.prop_cfg.get("obs_preproc", lambda obs: obs)
        self.obs_postproc = params.prop_cfg.get("obs_postproc", lambda obs, model_out: model_out)
        self.obs_postproc2 = params.prop_cfg.get("obs_postproc2", lambda next_obs: next_obs)
        self.targ_proc = params.prop_cfg.get("targ_proc", lambda obs, next_obs: next_obs)
        self.continue_train = params.prop_cfg.get("continue_train", False)

        self.opt_mode = get_required_argument(params.opt_cfg, "mode", "Must provide optimization method.")
        self.plan_hor = get_required_argument(params.opt_cfg, "plan_hor", "Must provide planning horizon.")
        self.obs_cost_fn = get_required_argument(params.opt_cfg, "obs_cost_fn", "Must provide cost on observations.")
        self.ac_cost_fn = get_required_argument(params.opt_cfg, "ac_cost_fn", "Must provide cost on actions.")
        self.catastrophe_cost_fn = get_required_argument(params.opt_cfg, "catastrophe_cost_fn", "Must provide cost on catastrophe.")
        self.no_catastrophe_pred = params.opt_cfg.get("no_catastrophe_pred")
        self.percentile = get_required_argument(params.opt_cfg, "percentile", "Must provide percentile used for optimizer")



        if hasattr(params.env, "possible_actions"):
            # Discrete Case
            self.possible_actions = params.env.possible_actions
        self.mode = 'train' #Setting mode to training or testing (adapting)
        
        assert self.prop_mode == 'TSinf', 'only TSinf propagation mode is supported'
        assert self.npart % self.model_init_cig.num_nets == 0, "Number of particles must be a multiple of the ensemble size."

        # Create action sequence optimizer
        opt_cfg = params.opt_cfg.get("cfg", {})
        optim_map = {'CEM': CEMOptimizer, 'DRO': DiscreteRandomOptimizer, 'DCEM': DiscreteCEMOptimizer}
        self.optimizer = optim_map[self.opt_mode](
            sol_dim=self.plan_hor * self.dU,
            lower_bound=np.tile(self.ac_lb, [self.plan_hor]),
            upper_bound=np.tile(self.ac_ub, [self.plan_hor]),
            cost_function=self._compile_cost,
            **opt_cfg
        )

        # Controller state variables
        self.has_been_trained = params.prop_cfg.get("model_pretrained", False)
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])
        self.train_in = np.array([]).reshape(0, self.dU + self.obs_preproc(np.zeros([1, self.dO])).shape[-1])
        self.gravity_targs = np.array([]).reshape(0, 1)
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1]
        )

        print("Created an MPC controller, prop mode %s, %d particles. " % (self.prop_mode, self.npart) +
              ("Ignoring variance." if self.ign_var else ""))


        # Set up pytorch model
        self.model = get_required_argument(
            params.prop_cfg.model_init_cfg, "model_constructor", "Must provide a model constructor."
        )(params.prop_cfg.model_init_cfg)

        self.logdir = None

    def clear_buffers(self):
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])
        self.train_in = np.array([]).reshape(0, self.dU + self.obs_preproc(np.zeros([1, self.dO])).shape[-1])
        self.gravity_targs = np.array([]).reshape(0, 1)
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1]
        )


    def train(self, obs_trajs, acs_trajs, rews_trajs, gravity_vals=None):
        """Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.

        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            acs_trajs: A list of action matrices, actions in rows.
            rews_trajs: A list of reward arrays.

        Returns: None.
        """

        # Construct new training points and add to training set
        new_train_in, new_train_targs = [], []
        for obs, acs in zip(obs_trajs, acs_trajs):
            new_train_in.append(np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1))
            new_train_targs.append(self.targ_proc(obs[:-1], obs[1:]))
        self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
        self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)
        # Train the model
        self.has_been_trained = True

        # Train the pytorch model
        self.model.fit_input_stats(self.train_in)
        idxs = np.random.randint(self.train_in.shape[0], size=[self.model.num_nets, self.train_in.shape[0]])

        epochs = self.model_train_cfg['epochs']

        batch_size = 256 if 'batch_size' not in self.model_train_cfg else self.model_train_cfg['batch_size']

        epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))
        self.mse_loss = None
        self.catastrophe_loss = None
        self.epochs = epochs
        for _ in epoch_range:

            for batch_num in range(num_batch):
                batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]

                loss = 0.01 * (self.model.max_logvar.sum() - self.model.min_logvar.sum())
                loss += self.model.compute_decays()

                train_in = torch.from_numpy(self.train_in[batch_idxs]).to(TORCH_DEVICE).float()
                train_targ = torch.from_numpy(self.train_targs[batch_idxs]).to(TORCH_DEVICE).float()
                state_targ = train_targ[..., :-1]
                catastrophe_targ = train_targ[..., -1:]
                mean, logvar, catastrophe_prob = self.model(train_in, ret_logvar=True)
                inv_var = torch.exp(-logvar)
                state_loss = ((mean - state_targ) ** 2) * inv_var + logvar
                state_loss = state_loss.mean(-1).mean(-1).sum()
                if not self.no_catastrophe_pred:
                    num_catastrophes = torch.sum(catastrophe_targ == 1)
                    if num_catastrophes == 0:
                        pos_weight = 0 * catastrophe_targ[0][0]
                    else:
                        pos_weight = (catastrophe_targ.numel() - num_catastrophes).to(dtype=torch.float) / num_catastrophes
                    catastrophe_loss = torch.nn.BCEWithLogitsLoss(pos_weight)(catastrophe_prob, catastrophe_targ)
                    loss += catastrophe_loss
                loss += state_loss
                self.model.optim.zero_grad()
                loss.backward()
                self.model.optim.step()
            idxs = shuffle_rows(idxs)

            #Print loss on 5000 random samples
            with torch.no_grad():
                val_in = torch.from_numpy(self.train_in[idxs[:, :5000]]).to(TORCH_DEVICE).float()
                val_targ = torch.from_numpy(self.train_targs[idxs[:, :5000]]).to(TORCH_DEVICE).float()
                val_state_targ = val_targ[..., :-1]
                val_catastrophe_targ = val_targ[..., -1:]
                mean, _, catastrophe_prob = self.model(val_in)
                mse_losses = ((mean - val_state_targ) ** 2).mean(-1).mean(-1)
                if not self.no_catastrophe_pred:
                    num_catastrophes = torch.sum(val_catastrophe_targ == 1)
                    if num_catastrophes == 0:
                        pos_weight = 0 * val_catastrophe_targ[0][0]
                    else:
                        pos_weight = (val_catastrophe_targ.numel() - num_catastrophes).to(dtype=torch.float) / num_catastrophes
                    catastrophe_loss = torch.nn.BCEWithLogitsLoss(pos_weight)(catastrophe_prob, val_catastrophe_targ)
                    catastrophe_loss = catastrophe_loss.detach().cpu().numpy()
                    self.catastrophe_loss = catastrophe_loss
            mse_losses = mse_losses.detach().cpu().numpy()
            epoch_range.set_postfix({
                "State loss": mse_losses,
                "Catastrophe pred loss": catastrophe_loss if not self.no_catastrophe_pred else 0,
            })
            self.mse_loss = mse_losses

    def clear_stats(self):
        """Clears the tensors keeping track of statistics

        Returns: None
        """
        self.num_seen_so_far = 0


    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        if isinstance(self.optimizer, DiscreteRandomOptimizer):
            self.prev_sol = np.ones(shape=[self.plan_hor]) * (1 / self.plan_hor)
        elif isinstance(self.optimizer, DiscreteCEMOptimizer):
            self.prev_sol = np.ones(shape = [self.plan_hor, self.dU]) * (1 / self.dU)
        else:
            self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.optimizer.reset()
        for update_fn in self.update_fns:
            update_fn()

    def act(self, obs, t):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep

        Returns: An action (and possibly the predicted cost)
        """
        d_random = isinstance(self.optimizer, DiscreteRandomOptimizer)
        d_cem = isinstance(self.optimizer, DiscreteCEMOptimizer)
        cem = isinstance(self.optimizer, CEMOptimizer)
        if d_random or d_cem:
            if not self.has_been_trained:
                return self.possible_actions[np.random.choice(np.arange(self.possible_actions.shape[-1]), size=1)[0]]
            if self.ac_buf.shape[0] > 0:
                action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
                if d_random:
                    return action
                return self.possible_actions[np.argmax(action)]
            self.sy_cur_obs = obs
            soln = self.optimizer.obtain_solution(self.prev_sol, self.possible_actions)
            if d_random:
                self.prev_sol = np.concatenate([np.copy(soln)[self.per * self.dU:], np.zeros(self.per * self.dU)])
                self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)
            elif d_cem:
                self.prev_sol = np.concatenate([np.copy(soln)[1:], np.zeros((1, self.per * self.dU))])
                self.ac_buf = soln[:1].reshape(-1, self.dU)

            return self.act(obs, t)

        else:
            if not self.has_been_trained:
                return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
            if self.ac_buf.shape[0] > 0:
                action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
                return action

            self.sy_cur_obs = obs

            soln = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
            self.prev_sol = np.concatenate([np.copy(soln)[self.per * self.dU:], np.zeros(self.per * self.dU)])
            self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)

            return self.act(obs, t)


    @torch.no_grad()
    def _compile_cost(self, ac_seqs):
        nopt = ac_seqs.shape[0]
        ac_seqs = torch.from_numpy(ac_seqs).float().to(TORCH_DEVICE)

        # Reshape ac_seqs so that it's amenable to parallel compute
        ac_seqs = ac_seqs.view(-1, self.plan_hor, self.dU)
        transposed = ac_seqs.transpose(0, 1)
        expanded = transposed[:, :, None]
        tiled = expanded.expand(-1, -1, self.npart, -1)
        ac_seqs = tiled.contiguous().view(self.plan_hor, -1, self.dU)

        # Expand current observation
        cur_obs = torch.from_numpy(self.sy_cur_obs).float().to(TORCH_DEVICE)
        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(nopt * self.npart, -1)
        costs = torch.zeros(nopt, self.npart, device=TORCH_DEVICE)
        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]
            next_obs = self._predict_next_obs(cur_obs, cur_acs)
            cost = self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs)
            if self.mode == 'test' and not self.no_catastrophe_pred: #use catastrophe prediction during adaptation
                cost = self.catastrophe_cost_fn(next_obs, cost, self.percentile)
            cost = cost.view(-1, self.npart)
            costs += cost
            cur_obs = self.obs_postproc2(next_obs)
        # replace nan with high cost
        costs[costs != costs] = 1e6
        if self.no_catastrophe_pred:
            # Discounted reward sum calculation for CARL (Reward). At self.percentile == 100, this is normal PETS
            if self.percentile <= 100:
                k_percentile = -(-costs).kthvalue(k=max(int((self.percentile/100) * costs.shape[1]), 1), dim=1)[0]
                cost_mask = costs <  k_percentile.view(-1, 1).repeat(1, costs.shape[1])
            else:
                k_percentile = costs.kthvalue(k=max(int(((200 - self.percentile)/100) * costs.shape[1]), 1), dim=1)[0]
                cost_mask = costs >  k_percentile.view(-1, 1).repeat(1, costs.shape[1])
            costs[cost_mask] = 0
            discounted_sum = costs.sum(dim=1)
            costs[cost_mask] = float('nan')
            lengths = torch.sum(~torch.isnan(costs), dim=1).float()
            mean_cost = discounted_sum / lengths
        else:
            mean_cost = costs.mean(dim=1)
        return mean_cost.detach().cpu().numpy()

    def _predict_next_obs(self, obs, acs):
        proc_obs = self.obs_preproc(obs)

        assert self.prop_mode == 'TSinf'
        proc_obs = self._expand_to_ts_format(proc_obs)
        acs = self._expand_to_ts_format(acs)

        inputs = torch.cat((proc_obs, acs), dim=-1)

        mean, var, catastrophe_prob = self.model(inputs)

        predictions = mean + torch.randn_like(mean, device=TORCH_DEVICE) * var.sqrt()

        predictions = torch.cat((predictions, catastrophe_prob), dim=-1)

        # TS Optimization: Remove additional dimension
        predictions = self._flatten_to_matrix(predictions)

        return self.obs_postproc(obs, predictions)


    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]

        reshaped = mat.view(-1, self.model.num_nets, self.npart // self.model.num_nets, dim)

        transposed = reshaped.transpose(0, 1)

        reshaped = transposed.contiguous().view(self.model.num_nets, -1, dim)

        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]

        reshaped = ts_fmt_arr.view(self.model.num_nets, -1, self.npart // self.model.num_nets, dim)

        transposed = reshaped.transpose(0, 1)

        reshaped = transposed.contiguous().view(-1, dim)

        return reshaped

