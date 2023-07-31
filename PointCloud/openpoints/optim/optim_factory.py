""" Optimizer Factory w/ Custom Weight Decay
Borrowed from Ross Wightman (https://www.github.com/timm)
"""
from typing import Optional
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import json

from .adabelief import AdaBelief
from .adafactor import Adafactor
from .adahessian import Adahessian
from .adamp import AdamP
from .lamb import Lamb
from .lars import Lars
from .lookahead import Lookahead
from .madgrad import MADGRAD
from .nadam import Nadam
from .nvnovograd import NvNovoGrad
from .radam import RAdam
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP

import logging

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD

    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    # remove module, and encoder.
    var_name = copy.deepcopy(var_name)
    var_name = var_name.replace('module.', '')
    var_name = var_name.replace('encoder.', '')

    if any(key in var_name for key in {"cls_token", "mask_token", "cls_pos", "pos_embed", "patch_embed"}):
        return 0
    elif "rel_pos_bias" in var_name:
        return num_max_layer - 1

    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, 
                         filter_by_modules_names=None, 
                         ):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias") or any(key in name for key in skip_list):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None
        
        if get_layer_scale is not None:
            scale = get_layer_scale(layer_id) * scale
        else:
            scale = 1.0
        
        if filter_by_modules_names is not None:
            filter_exist = False 
            for module_name in filter_by_modules_names.keys():
                filter_exist = module_name in name
                if filter_exist:
                    break 
            if filter_exist:
                group_name = module_name + '_' + group_name
                this_weight_decay = filter_by_modules_names[module_name].get('weight_decay', this_weight_decay)
                scale = filter_by_modules_names[module_name].get('lr_scale', 1.0) * scale

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    logging.info("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        opt=cfg.opt,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs


def build_optimizer_from_cfg(
        model,
        NAME: str = 'sgd',
        lr: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        filter_bias_and_bn: bool = True,
        filter_by_modules_names=None, 
        **kwargs):
    """ Create an optimizer.
    Args:
        model (nn.Module): model containing parameters to optimize
        NAME: name of optimizer to create
        lr: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through
    Returns:
        Optimizer
    """
    # layer lr decay
    layer_decay = kwargs.get('layer_decay', 0)
    if 0. < layer_decay < 1.0:
        num_layers = model.get_num_layers()
        assigner = LayerDecayValueAssigner(
            list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        get_num_layer = assigner.get_layer_id
        get_layer_scale = assigner.get_scale
    else:
        get_num_layer, get_layer_scale = None, None

    assert isinstance(model, nn.Module)
    # a model was passed in, extract parameters and add weight decays to appropriate layers
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'module'):
            if hasattr(model.module, 'no_weight_decay'):
                skip = model.module.no_weight_decay()
        else:
            if hasattr(model, 'no_weight_decay'):
                skip = model.module.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, 
                                          filter_by_modules_names
                                          )
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_lower = NAME.lower()
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(weight_decay=weight_decay, **kwargs)
    if lr is not None:
        opt_args.setdefault('lr', lr)

    # basic SGD & related
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        # NOTE 'sgd' refers to SGD + nesterov momentum for legacy / backwards compat reasons
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=momentum, nesterov=True, **opt_args)

    # adaptive
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'nadam':
        try:
            # NOTE PyTorch >= 1.10 should have native NAdam
            optimizer = optim.Nadam(parameters, **opt_args)
        except AttributeError:
            optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamax':
        optimizer = optim.Adamax(parameters, **opt_args)
    elif opt_lower == 'adabelief':
        optimizer = AdaBelief(parameters, rectify=False, **opt_args)
    elif opt_lower == 'radabelief':
        optimizer = AdaBelief(parameters, rectify=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adagrad':
        opt_args.setdefault('eps', 1e-8)
        optimizer = optim.Adagrad(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'lamb':
        optimizer = Lamb(parameters, **opt_args)
    elif opt_lower == 'lambc':
        optimizer = Lamb(parameters, trust_clip=True, **opt_args)
    elif opt_lower == 'larc':
        optimizer = Lars(parameters, momentum=momentum, trust_clip=True, **opt_args)
    elif opt_lower == 'lars':
        optimizer = Lars(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'nlarc':
        optimizer = Lars(parameters, momentum=momentum, trust_clip=True, nesterov=True, **opt_args)
    elif opt_lower == 'nlars':
        optimizer = Lars(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'madgrad':
        optimizer = MADGRAD(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'madgradw':
        optimizer = MADGRAD(parameters, momentum=momentum, decoupled_decay=True, **opt_args)
    elif opt_lower == 'novograd' or opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=momentum, **opt_args)

    # second order
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)

    # NVIDIA fused optimizers, require APEX to be installed
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)

    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
