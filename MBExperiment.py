from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from time import localtime, strftime, perf_counter

from dotmap import DotMap
from scipy.io import savemat
from tqdm import trange

from Agent import Agent
from DotmapUtils import get_required_argument
import pickle
from tensorboardX import SummaryWriter
import numpy as np

from gym import wrappers
import torch

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MBExperiment:
    def __init__(self, params):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment.
                    .task_hor (int): Task horizon.
                    .test_percentile (float): Risk-aversion percentile used for testing.
                    .record_video (bool): Whether to record training/adaptation iterations.

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.
                    .ntest_rollouts (int): Number of rollouts for measuring test performance.
                    .nadapt_iters (int): (optional) Number of adaptation iters to perform. 10 in paper.
                    .continue_train (bool): Whether to continue training from a load_model_dir.
                    .test_domain (float): Environment domain used for adaptation/testing.
                    .nrollout_per_itr (int): Number of rollouts per training iteration.
                    .start_epoch (int): Which epoch to start training from, used for continuing to train
                        a trained model.

                .log_cfg:
                    .logdir (str): Directory to log to.
                    .suffix (str): Suffix to add to logdir.


        """

        # Assert True arguments that we currently do not support
        assert params.sim_cfg.get("stochastic", False) == False

        self.env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )
        self.test_percentile = params.sim_cfg.test_percentile
        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        self.ninit_rollouts = params.exp_cfg.get("ninit_rollouts", 1)
        self.ntest_rollouts = params.exp_cfg.get("ntest_rollouts", 1)
        self.nadapt_iters = params.exp_cfg.get("nadapt_iters", 0)
        self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")
        self.continue_train = params.exp_cfg.get("continue_train", False)
        self.test_domain = params.exp_cfg.get("test_domain", None)
        self.nrollout_per_itr = params.exp_cfg.get("nrollout_per_itr", 1)
        self.start_epoch = params.exp_cfg.get("start_epoch", 0)
        
        self.training_percentile = self.policy.percentile

        if self.continue_train:
            self.logdir = params.exp_cfg.load_model_dir
            self.policy.ac_buf = np.load(os.path.join(self.logdir, "ac_buf.npy"))
            self.policy.prev_sol = np.load(os.path.join(self.logdir, "prev_sol.npy"))
            self.policy.init_var = np.load(os.path.join(self.logdir, "init_var.npy"))
            self.policy.train_in = np.load(os.path.join(self.logdir, "train_in.npy"))
            self.policy.train_targs = np.load(os.path.join(self.logdir, "train_targs.npy"))
        self.logdir = os.path.join(
            get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory."),
            strftime("%Y-%m-%d--%H-%M-%S", localtime())
        )
        self.suffix = params.log_cfg.get("suffix", None)
        if self.suffix is not None:
            self.logdir = self.logdir + '-' + self.suffix
        self.writer = SummaryWriter(self.logdir + '-tboard')

        self.record_video = params.sim_cfg.get("record_video", False)
        if self.test_domain is not None:
            self.env.test_domain = self.test_domain
            print("Setting test domain to: %0.3f" % self.env.test_domain)

    def run_experiment(self):
        """Perform experiment.
        """
        os.makedirs(self.logdir, exist_ok=True)

        # Train with random data first
        samples = []
        self.agent = Agent()
        for i in range(self.ninit_rollouts):
            if self.record_video:
                self.record_env = wrappers.Monitor(self.env, "%s/init_iter_%d" % (self.logdir, i), force=True)
            samples.append(
                self.agent.sample(
                    self.task_hor, self.policy, record=False,
                    env=self.env,
                )
            )
        if self.ninit_rollouts > 0:
            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples],
            )

        self.run_training_iters(adaptation=False)

        # Save training buffers at end of training so we can load for adaptation if required
        old_train_in = self.policy.train_in
        old_train_targs = self.policy.train_targs
        old_ac_buf = self.policy.ac_buf
        old_prev_sol = self.policy.prev_sol
        old_init_var = self.policy.init_var
        torch.save(self.policy.model.state_dict(),
                os.path.join(self.logdir, 'weights'))
        np.save(os.path.join(self.logdir, "ac_buf.npy"), old_ac_buf)
        np.save(os.path.join(self.logdir, "prev_sol.npy"), old_prev_sol)
        np.save(os.path.join(self.logdir, "init_var.npy"), old_init_var)
        np.save(os.path.join(self.logdir, "train_in.npy"), old_train_in)
        np.save(os.path.join(self.logdir, "train_targs.npy"), old_train_targs)
        
        self.run_training_iters(adaptation=True)
        self.run_test_evals(self.nadapt_iters)

    def run_training_iters(self, adaptation):
        max_return = -float("inf")
        if adaptation:
            iteration_range = [self.nadapt_iters]
            percentile = self.test_percentile
            print_str = "ADAPT"
        else:
            iteration_range = [self.start_epoch, self.ntrain_iters]
            percentile = self.training_percentile
            print_str = "TRAIN"
        for i in trange(*iteration_range):
            if i % 2 == 0 and adaptation:
                self.run_test_evals(i)
            print("####################################################################")
            print("Starting training on " + print_str + " env iteration %d" % (i + 1))

            samples = []
            self.policy.clear_stats()
            self.policy.percentile = percentile
            for j in range(max(self.nrollout_per_itr, self.nrollouts_per_iter)):
                self.policy.percentile = percentile
                if self.record_video:
                    self.env = wrappers.Monitor(self.env, "%s/%s_iter_%d_percentile/percentile_%d_rollout_%d" % (self.logdir, print_str, i, self.policy.percentile, j), force=True)
                self.policy.logdir = "%s/%s_iter_%d" % (self.logdir, print_str, i)
                samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy, record=self.record_video and adaptation,
                        env=self.env, mode='test' if adaptation else 'train',
                    )
                )
            if self.record_video:
                self.env = self.env.env
            eval_samples = samples
            self.writer.add_scalar('mean-' + print_str + '-return',
                                   float(sum([sample["reward_sum"] for sample in eval_samples])) / float(len(eval_samples)),
                                   i)
            max_return = max(float(sum([sample["reward_sum"] for sample in eval_samples])) / float(len(eval_samples)), max_return)
            self.writer.add_scalar('max-' + print_str + '-return',
                                   max_return,
                                   i)
            rewards = [sample["reward_sum"] for sample in eval_samples]
            print("Rewards obtained:", rewards)
            samples = samples[:self.nrollouts_per_iter]

            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples],
            )
            if self.policy.mse_loss is not None:
                mean_loss = np.mean(self.policy.mse_loss)
                self.writer.add_scalar('%s-mean-loss' % print_str,
                                       mean_loss, i)
            if self.policy.catastrophe_loss is not None:
                self.writer.add_scalar('%s-catastrophe-loss' % print_str,
                                       self.policy.catastrophe_loss, i)

    def run_test_evals(self, adaptation_iteration):
        print("Beginning evaluation rollouts.")
        if self.test_percentile is not None:
            self.policy.percentile = self.test_percentile
        samples = []
        for i in range(self.ntest_rollouts):
            if self.record_video:
                self.env = wrappers.Monitor(self.env, "%s/test_eval_%d" % (self.logdir, i), force=True)
            if not hasattr(self, "agent"):
                self.agent = Agent()
            self.policy.clear_stats()
            cur_sample = self.agent.sample(
                    self.task_hor, self.policy, record=self.record_video,
                    env=self.env, mode='test',
                    )
            if self.record_video:
                self.env = self.env.env
            samples.append(cur_sample)
            mean_test_return = float(sum([cur_sample["reward_sum"] for sample in cur_sample])) / float(len(cur_sample))
            print("Evaluation mean-return (rollout number %d out of %d): %f" % (
                i,
                self.ntest_rollouts,
                mean_test_return
            ))
        if self.ntest_rollouts > 0:
            num_catastrophes = sum([1 if sample["catastrophe"] else 0 for sample in samples])
            self.writer.add_scalar('num-catastrophes',
                                   num_catastrophes,
                                   adaptation_iteration)
            mean_test_return = float(sum([sample["reward_sum"] for sample in samples])) / float(len(samples))
            self.writer.add_scalar('mean-test-return:',
                                   mean_test_return, adaptation_iteration)
        self.writer.close()