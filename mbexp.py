from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

from dotmap import DotMap

from MBExperiment import MBExperiment
from MPC import MPC
from config import create_config
import env # We run this so that the env is registered

import torch
import numpy as np
import random
import tensorflow as tf


def set_global_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    tf.set_random_seed(seed)


def main(args):
    #set_global_seeds(0)


    cfg = create_config(args)
    cfg.pprint()

    assert args.ctrl_type == 'MPC'

    cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
    exp = MBExperiment(cfg.exp_cfg)

    if args.load_model_dir is not None:
        exp.policy.model.load_state_dict(torch.load(os.path.join(args.load_model_dir, 'weights')))
    if not os.path.exists(exp.logdir):
        os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    exp.run_experiment()

    #torch.save(exp.policy.model.state_dict(),
    #        os.path.join(exp.logdir, 'weights'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True,
                        help='Environment name: select from [duckietown, cartpole, halfcheetahdisabled]')
    parser.add_argument('--test_domain', type=float,
                        help='The value for the test domain for the environment [eg. car width, pole length]. Not applicable to halfcheetahdisabled')
    parser.add_argument('--ninit_rollouts', type=int, default=None,
                        help='number of initial rollouts, defaults to 0 if loading pretrained model and arg not specified, 1 otherwise')
    parser.add_argument('--ntrain_iters', type=int, default=None,
                        help='number of training iterations, defaults to the one defined in config/[env].py')
    parser.add_argument('--nitr_per_rollout', type=int, default=0,
                        help='number of rollouts per training iteration')
    parser.add_argument('--nadapt_iters', type=int, default=10,
                        help='number of adaptation iterations to perform on test environment')
    parser.add_argument('--ntest_rollouts', type=int, default=1,
                        help='number of test rollouts to perform')
    parser.add_argument('--no_catastrophe_pred', action='store_true',
                         help='if this flag is set, disables training and utilizing catastrophe prediction. Should be enabled \
                             if training CARL (Reward) or MB + Finetune')
    parser.add_argument('--percentile', type=int, default=100,
                        help='caution parameter (gamma/beta)')
    parser.add_argument('--test_percentile', type=int, default=50,
                        help='caution parameter during test time (gamma/beta)')
    parser.add_argument('--num_nets', type=int, default=5,
                        help='number of networks in the ensemble')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='epoch to start training at, used in conjuction with -continue_train')
    parser.add_argument('--continue_train', action='store_true',
                        help='whether to continue training from the loaded model')
    parser.add_argument('--test_mode', action='store_true',
                        help='Will load model from directory pointed to by --logdir and set all training iteration flags to 0')
    parser.add_argument('--logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    parser.add_argument('--load_model_dir', type=str, default=None,
                        help='Directory from which weights will be loaded')
    parser.add_argument('--suffix', type=str, default=None,
                        help='suffix to attach to a run')
    parser.add_argument('--record_video', action='store_true',
                        help='whether to record the test rollouts')
    args = parser.parse_args()

    args.ctrl_type = "MPC"

    if args.start_epoch != 0:
        assert args.ntrain_iters > args.start_epoch, "must end at epoch greater than start epoch"
    if args.ntrain_iters != None and args.ntrain_iters > 0 and args.start_epoch != args.ntrain_iters and args.load_model_dir != None:
        args.continue_train = True
    if args.load_model_dir is not None and args.ninit_rollouts is None:
        args.ninit_rollouts = 0
    elif args.load_model_dir is None and args.ninit_rollouts is None:
        args.ninit_rollouts = 1
    if args.test_mode:
        args.ninit_rollouts = 0
        args.ntrain_iters = 0
    args.optimizer = 'CEM'
    main(args)
