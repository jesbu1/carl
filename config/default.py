from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dotmap import DotMap
import os
import importlib

def create_config(args):
    cfg = DotMap()
    type_map = DotMap(
        exp_cfg=DotMap(
            sim_cfg=DotMap(
                task_hor=int,
                stochastic=make_bool,
                noise_std=float,
                test_percentile=int,
                record_video=bool
            ),
            exp_cfg=DotMap(
                ntrain_iters=int,
                nrollouts_per_iter=int,
                ninit_rollouts=int,
                nadapt_iters=int,
                ntest_rollouts=int,
                continue_train=bool,
                load_model_dir=str,
                start_epoch=int,
                test_domain=float,
                nitr_per_rollout=int
            ),
            log_cfg=DotMap(
                suffix=str,
            )
        ),
        ctrl_cfg=DotMap(
            per=int,
            prop_cfg=DotMap(
                model_pretrained=make_bool,
                npart=int,
                ign_var=make_bool,
                model_init_cfg=DotMap()
            ),
            opt_cfg=DotMap(
                plan_hor=int,
                no_catastrophe_pred=bool,
                mode=str,
                percentile=int,
            ),
        )
    )
    ctrl_args = DotMap()
    has_been_trained = args.load_model_dir != None
    
    # This code seems to be for loading file based on args.env
    dir_path = os.path.dirname(os.path.realpath(__file__))
    loader = importlib.machinery.SourceFileLoader(args.env, os.path.join(dir_path, "%s.py" % args.env))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    cfg_source = importlib.util.module_from_spec(spec)
    loader.exec_module(cfg_source)
    cfg_module = cfg_source.CONFIG_MODULE()

    # cfg_module by here is an instance of <env>ConfigModule

    _create_exp_config(cfg.exp_cfg, cfg_module, type_map, args)
    cfg.exp_cfg.sim_cfg.test_percentile = args.test_percentile
    cfg.exp_cfg.sim_cfg.record_video = args.record_video

    _create_ctrl_config(cfg.ctrl_cfg, cfg_module, args.ctrl_type, ctrl_args, type_map, has_been_trained, args.optimizer, args.percentile, args.num_nets, args.no_catastrophe_pred)
    

    return cfg


def _create_exp_config(exp_cfg, cfg_module, type_map, args):
    exp_cfg.sim_cfg.env = cfg_module.ENV
    exp_cfg.sim_cfg.task_hor = cfg_module.TASK_HORIZON
    
    exp_cfg.exp_cfg.ntrain_iters = args.ntrain_iters if args.ntrain_iters is not None else cfg_module.NTRAIN_ITERS
    exp_cfg.exp_cfg.nadapt_iters = args.nadapt_iters
    exp_cfg.exp_cfg.nrollouts_per_iter = cfg_module.NROLLOUTS_PER_ITER
    exp_cfg.exp_cfg.ntest_rollouts = args.ntest_rollouts if args.ntest_rollouts is not None else cfg_module.NTEST_ROLLOUTS
    exp_cfg.exp_cfg.ninit_rollouts = args.ninit_rollouts
    exp_cfg.exp_cfg.continue_train = args.continue_train
    exp_cfg.exp_cfg.load_model_dir = args.load_model_dir
    exp_cfg.exp_cfg.start_epoch = args.start_epoch
    exp_cfg.exp_cfg.test_domain = args.test_domain

    exp_cfg.log_cfg.logdir = args.logdir
    exp_cfg.log_cfg.suffix = args.suffix
    exp_cfg.log_cfg.nitr_per_rollout = args.nitr_per_rollout


def _create_ctrl_config(ctrl_cfg, cfg_module, ctrl_type, ctrl_args, type_map, has_been_trained, optimizer, percentile, num_nets, no_catastrophe_pred):
    """Creates controller configuration.

    """
    assert ctrl_type == 'MPC'

    ctrl_cfg.env = cfg_module.ENV
    if hasattr(cfg_module, "UPDATE_FNS"):
        ctrl_cfg.update_fns = cfg_module.UPDATE_FNS
    if hasattr(cfg_module, "obs_preproc"):
        ctrl_cfg.prop_cfg.obs_preproc = cfg_module.obs_preproc
    if hasattr(cfg_module, "obs_postproc"):
        ctrl_cfg.prop_cfg.obs_postproc = cfg_module.obs_postproc
    if hasattr(cfg_module, "targ_proc"):
        ctrl_cfg.prop_cfg.targ_proc = cfg_module.targ_proc

    ctrl_cfg.opt_cfg.plan_hor = cfg_module.PLAN_HOR
    ctrl_cfg.opt_cfg.no_catastrophe_pred = no_catastrophe_pred
    ctrl_cfg.opt_cfg.obs_cost_fn = cfg_module.obs_cost_fn
    ctrl_cfg.opt_cfg.ac_cost_fn = cfg_module.ac_cost_fn
    ctrl_cfg.opt_cfg.catastrophe_cost_fn = cfg_module.catastrophe_cost_fn

    # Process arguments here.
    model_init_cfg = ctrl_cfg.prop_cfg.model_init_cfg

    # Set model class
    ctrl_args["model-type"] = 'PE'
    model_init_cfg.num_nets = num_nets

    ctrl_cfg.prop_cfg.model_train_cfg = cfg_module.NN_TRAIN_CFG
    model_init_cfg.model_constructor = cfg_module.nn_constructor

    # Add possible overrides
    type_map.ctrl_cfg.prop_cfg.model_init_cfg.model_dir = str
    type_map.ctrl_cfg.prop_cfg.model_init_cfg.load_model = make_bool

    type_map.ctrl_cfg.prop_cfg.model_train_cfg = DotMap(
        batch_size=int, epochs=int,
        holdout_ratio=float, max_logging=int
    )

    ctrl_cfg.prop_cfg.mode = "TSinf"
    ctrl_cfg.prop_cfg.npart = 20
    ctrl_cfg.prop_cfg.model_pretrained = has_been_trained
    # Finish setting model class

    # Setting MPC cfg
    ctrl_cfg.opt_cfg.mode = optimizer
    ctrl_cfg.opt_cfg.percentile = percentile
    type_map.ctrl_cfg.opt_cfg.cfg = DotMap(
        max_iters=int,
        popsize=int,
        num_elites=int,
        epsilon=float,
        alpha=float
    )
    ctrl_cfg.opt_cfg.cfg = cfg_module.OPT_CFG[ctrl_cfg.opt_cfg.mode]


def apply_override(cfg, type_map, override_key, value, prefix=''):
    """Modifies the configuration to apply the given override.
    """
    pth = override_key.split(".")
    filter_pth = prefix.split(".")
    if len(prefix) == 0 or pth[:len(filter_pth)] == prefix.split("."):
        cur_map = cfg
        cur_type_map = type_map
        try:
            for key in pth[:-1]:
                cur_map = cur_map[key]
                cur_type_map = cur_type_map[key]
        except KeyError:
            raise KeyError(
                "Either %s cannot be overridden (is a function/object/class/etc.) or "
                "the type map is not updated." % override_key
            )
        if cur_type_map.get(pth[-1], None) is None:
            raise KeyError(
                "Either %s cannot be overridden (is a function/object/class/etc.) or "
                "the type map is not updated." % override_key
            )
        cur_map[pth[-1]] = cur_type_map[pth[-1]](value)

    return cfg

def make_bool(arg):
    if arg == "False" or arg == "false" or not bool(arg):
        return False
    else:
        return True

def make_bool(arg):
    if arg == "False" or arg == "false" or not bool(arg):
        return False
    else:
        return True


def create_read_only(message):
    def read_only(arg):
        raise RuntimeError(message)

    return read_only


def create_conditional(cl, cond, message):
    def conditional(arg):
        if cond(arg):
            return cl(arg)
        else:
            raise RuntimeError(message)

    return conditional
