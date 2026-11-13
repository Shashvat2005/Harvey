from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval

import numpy as np

"""config system.
This file specifies default config options. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.
"""


class AttrDict(dict):
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value


__C = AttrDict()

cfg = __C

# Check the training setting for each method for original paper
__C.METHOD = 'cross_entropy'
__C.DATASET = 'cifar10'
__C.DATA_ROOT = './data'
__C.TASK_LEARNER = 'wrn161'
__C.REF_LEARNER = 'wrn161'
__C.FULLSET = False #if True, use SUBSET in each query interation
__C.TOTAL_DATASET = False #if True, labeled all data points initially and train them at once
__C.CYCLES = 18 
__C.TRIALS = 5
__C.INIT_BUDGET = 1000
__C.BUDGET = 1000 
__C.SUB_PRE = 10000 #num of subset
__C.BATCH = 128 # general batch size for data loader
__C.NUM_TRAIN = 50000 # size of the original data pool
__C.RANDOM_SET_SEED = None
__C.RANDOM_SET_SEED_ORDER = None
__C.RANDOM_MODEL_PARAM_SEED = None
__C.BUDGET = 1000
__C.STRAT_SAMPLING = False # use stratified sampling for initially labeled pool, useful for INIT_BUDGET=10
__C.DATASET_NOISE_STD = 0.2 # standard deviation of gaussian noise in cifar10noise dataset
__C.DATASET_NOISE_PORTION = 0.8 # portion of noisy samples in cifar10noise dataset
__C.DATASET_IMB_RATIO = 1 # ratio between number of samples in normal and reduced classes in cifar10im dataset
__C.results_folder = 'results_paper/batch_size_1/'

__C.init_ratio = 0.5
__C.min_ratio = 0.05
__C.cls_min_ratio = 0.01  # the lowest preservation ratio of samples in each class
__C.warm_start = True
__C.ref_warm_start = True
__C.second_stg = 'SCE'
__C.ref_stg = 'SCE'
__C.LGA = False
__C.LGA_gamma = 0.01
__C.unlearn_ref = False
__C.unlearn_gamma = 0.0001
__C.unlearn_epochs = 1  # legacy GA baseline only
__C.UNLEARNING = 'CATA'  # CATA or GA (original HARVEY ablation)
__C.CATA = AttrDict()
__C.CATA.KEEP_RATIO = 0.3
__C.CATA.SCALE = 0.7
__C.CATA.EPOCHS = 4
__C.CATA.LR = 1e-5
__C.CATA.WEIGHT_DECAY = 0.1

__C.plot_hist = False
__C.plot_tsne = False

__C.print_cls_dist = False

__C.GCN = AttrDict()
__C.GCN.DROPOUT_RATE = 0.3
__C.GCN.S_MARGIN = 0.1
__C.GCN.HIDDEN_UNITS = 128
__C.GCN.LAMBDA_LOSS = 1.2
__C.GCN.EPOCH_GCN = 200
__C.GCN.LR_GCN = 1e-3
__C.GCN.WDECAY_GCN = 5e-4

__C.VAAL = AttrDict() 
__C.VAAL.EPOCHV = 100 # training epochs for vae and diminator in VAAL, TA-VAAL, Inf-VAAL
__C.VAAL.Z_DIM = 32
__C.VAAL.LR = 5e-4

__C.BALD = AttrDict()
__C.BALD.DROPOUT_ITER = 25


__C.OPTIMIZER_BACKBONE = AttrDict()
__C.OPTIMIZER_BACKBONE.NAME = 'SGD'
__C.OPTIMIZER_BACKBONE.NAME_REF = 'SGD'
__C.OPTIMIZER_BACKBONE.MOMENTUM =  0.9
__C.OPTIMIZER_BACKBONE.WDECAY = 5e-4
__C.OPTIMIZER_BACKBONE.LR = 1e-1
__C.OPTIMIZER_BACKBONE.LR_REF = 1e-3
__C.OPTIMIZER_BACKBONE.MILESTONES = [50, 80]
__C.OPTIMIZER_BACKBONE.EPOCH_TL = 100 # train epoch for task learner

__C.OPTIMIZER_MODULE = AttrDict()
__C.OPTIMIZER_MODULE.NAME = 'SGD'
__C.OPTIMIZER_MODULE.MOMENTUM =  0.9
__C.OPTIMIZER_MODULE.WDECAY = 1e-4
__C.OPTIMIZER_MODULE.LR = 1e-1 #the original changes are 1e-2
__C.OPTIMIZER_MODULE.MILESTONES = [50, 80]
__C.OPTIMIZER_MODULE.MARGIN = 1.0 #xi
__C.OPTIMIZER_MODULE.WEIGHT = 1.0 #lambda
__C.OPTIMIZER_MODULE.EPOCHL =100 #for lloss and TA-VAAL, after 120 epoch should add the predicted loss for training

# SSL
__C.SSL = AttrDict()
__C.SSL.lambda_u = 10
__C.SSL.sl_train = False
__C.SSL.finetune = False

# BACKDOOR (Attacks)
__C.BACKDOOR = AttrDict()
__C.BACKDOOR.inject_portion = 0.1
__C.BACKDOOR.target_label = 5
__C.BACKDOOR.trigger_type = 'badnetsTrigger'
__C.BACKDOOR.target_type = 'all2one'
__C.BACKDOOR.trig_w = 3
__C.BACKDOOR.trig_h = 3
__C.BACKDOOR.exclude_target = False

# BACKDOOR (Defense)
__C.BACKDOOR.warmup_epochs = 10
__C.BACKDOOR.ref_cycle = 5
__C.BACKDOOR.ref_epochs = 20
__C.BACKDOOR.ref_c1epochs = 10
__C.BACKDOOR.split_rate = 0.2
__C.BACKDOOR.reinit_ref = True
__C.BACKDOOR.ref_init_train = False
__C.BACKDOOR.use_init_thresh = True
__C.BACKDOOR.meta_split = True

# Dynamic Attack
__C.BACKDOOR.input_channel = 3

# D-duplicate attack
__C.BACKDOOR.dduplicate = False


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), "Argument `a` must be an AttrDict"
    assert isinstance(b, AttrDict), "Argument `b` must be an AttrDict"

    for k, v_ in a.items():
        full_key = ".".join(stack) + "." + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("Non-existent config key: {}".format(full_key))

        v = _decode_cfg_value(v_)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v



def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml

    with open(filename, "r") as f:
        yaml_cfg = AttrDict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)
 
    return cfg


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
  
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_a, int) and (value_b is None):
        value_a = value_a
    else:
        raise ValueError(
            "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
            "key: {}".format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
