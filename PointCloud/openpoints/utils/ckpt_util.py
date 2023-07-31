import logging
import os, shutil
from termcolor import colored
from typing import Any
from typing import Optional, List, Dict, NamedTuple, Tuple, Iterable
from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn


# ================ model related ==================
def cal_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


def resume_model(model, cfg, pretrained_path=None):
    pretrained_path = os.path.join(cfg.ckpt_dir, os.path.join(cfg.run_name,
                                                              '_ckpt_latest.pth')) if pretrained_path is None else pretrained_path
    if not os.path.exists(pretrained_path):
        logging.info(f'[RESUME INFO] no checkpoint file from path {pretrained_path}...')
        return 0, 0
    logging.info(f'[RESUME INFO] Loading model weights from {pretrained_path}...')

    # load state dict
    state_dict = torch.load(pretrained_path, map_location='cpu')
    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    model.load_state_dict(base_ckpt, strict=True)

    # parameter
    if 'epoch' in state_dict.keys():
        start_epoch = state_dict['epoch'] + 1
    else:
        start_epoch = 1
    if 'best_metrics' in state_dict.keys():
        best_metrics = state_dict['best_metrics']
        if not isinstance(best_metrics, dict):
            best_metrics = best_metrics.state_dict()
    else:
        best_metrics = None

    logging.info(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})')
    return start_epoch, best_metrics


def resume_optimizer(cfg, optimizer, pretrained_path=None):
    pretrained_path = os.path.join(cfg.ckpt_dir, os.path.join(cfg.run_name,
                                                              '_ckpt_latest.pth')) if pretrained_path is None else pretrained_path
    if not os.path.exists(pretrained_path):
        logging.info(f'[RESUME INFO] no checkpoint file from path {pretrained_path}...')
        return 0, 0, 0
    logging.info(f'[RESUME INFO] Loading optimizer from {pretrained_path}...')
    # load state dict
    state_dict = torch.load(pretrained_path, map_location='cpu')
    # optimizer
    if state_dict['optimizer'] is not None and state_dict['optimizer']:
        optimizer.load_state_dict(state_dict['optimizer'])


def save_checkpoint(cfg, model, epoch, optimizer=None, scheduler=None,
                    additioanl_dict=None,
                    is_best=False, post_fix='ckpt_latest', save_name=None, ):
    if save_name is None:
        save_name = cfg.run_name

    current_ckpt_name = f'{save_name}_{post_fix}.pth'
    current_pretrained_path = os.path.join(cfg.ckpt_dir, current_ckpt_name)
    save_dict = {
        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else dict(),
        'epoch': epoch
    }
    if additioanl_dict is not None:
        save_dict.update(additioanl_dict)

    torch.save(save_dict, current_pretrained_path)

    if cfg.save_freq > 0 and epoch % cfg.save_freq == 0:
        milestone_ckpt_name = f'{save_name}_E{epoch}.pth'
        milestone_pretrained_path = os.path.join(cfg.ckpt_dir, milestone_ckpt_name)
        shutil.copyfile(current_pretrained_path, milestone_pretrained_path)
        logging.info("Saved in {}".format(milestone_pretrained_path))

    if is_best:
        best_ckpt_name = f'{save_name}_ckpt_best.pth' if save_name else 'ckpt_best.pth'
        best_pretrained_path = os.path.join(cfg.ckpt_dir, best_ckpt_name)
        shutil.copyfile(current_pretrained_path, best_pretrained_path)
        logging.info("Found the best model and saved in {}".format(best_pretrained_path))


def resume_checkpoint(config, model, optimizer=None, scheduler=None, pretrained_path=None, printer=logging.info):
    if pretrained_path is None:
        pretrained_path = config.pretrained_path
        assert pretrained_path is not None
    printer("=> loading checkpoint '{}'".format(pretrained_path))

    checkpoint = torch.load(pretrained_path, map_location='cpu')
    config.start_epoch = checkpoint['epoch'] + 1
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            printer('optimizer does not match')
    if scheduler is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            printer('scheduler does not match')

    ckpt_state = checkpoint['model']
    model_dict = model.state_dict()
    # rename ckpt (avoid name is not same because of multi-gpus)
    is_model_multi_gpus = True if list(model_dict)[0].split('.')[0] == 'module' else False
    is_ckpt_multi_gpus = True if list(ckpt_state)[0].split('.')[0] == 'module' else False

    if not (is_model_multi_gpus == is_ckpt_multi_gpus):
        temp_dict = OrderedDict()
        for k, v in ckpt_state.items():
            if is_ckpt_multi_gpus:
                name = k[7:]  # remove 'module.'
            else:
                name = 'module.' + k  # add 'module'
            temp_dict[name] = v
        ckpt_state = temp_dict

    model.load_state_dict(ckpt_state)

    config.start_epoch = checkpoint['epoch'] + 1
    config.epoch = checkpoint['epoch'] + 1
    printer("=> loaded successfully '{}' (epoch {})".format(pretrained_path, checkpoint['epoch']))
    del checkpoint
    torch.cuda.empty_cache()


def load_checkpoint(model, pretrained_path, module=None):
    if not os.path.exists(pretrained_path):
        raise NotImplementedError('no checkpoint file from path %s...' % pretrained_path)
    # load state dict
    state_dict = torch.load(pretrained_path, map_location='cpu')

    # parameter resume of base model
    ckpt_state_dict = state_dict
    for key in state_dict.keys():
        if key in ['model', 'net', 'network', 'state_dict', 'base_model']:
            ckpt_state_dict = ckpt_state_dict[key]
    base_ckpt = {k.replace("module.", ""): v for k, v in ckpt_state_dict.items()}
    if module is not None:
        base_ckpt = {k:v for k, v in base_ckpt.items() if module in k}
        
    if "bert" in list(ckpt_state_dict.items())[0][0]:
        base_ckpt=bert2vit_ckpt_rename(ckpt_state_dict)
        #state_dict has "qkv.value" key, will be mis-regonized as metric and over flush the command ouput
        state_dict=base_ckpt
         
    if hasattr(model, 'module'):
        incompatible = model.module.load_state_dict(base_ckpt, strict=False)
    else:
        incompatible = model.load_state_dict(base_ckpt, strict=False)
    if incompatible.missing_keys:
        logging.info('missing_keys')
        logging.info(
            get_missing_parameters_message(incompatible.missing_keys),
        )
    if incompatible.unexpected_keys:
        logging.info('unexpected_keys')
        logging.info(
            get_unexpected_parameters_message(incompatible.unexpected_keys)
        )
    logging.info(f'Successful Loading the ckpt from {pretrained_path}')

    epoch = state_dict.get('epoch', -1)
    metrics = {}
    for key in state_dict.keys():
        is_metric_key = sum([item in key for item in ['metric', 'acc', 'test', 'val']]) > 0
        if is_metric_key:
            metrics[key] = state_dict[key]
    logging.info(f'ckpts @ {epoch} epoch( {metrics} )')
    return epoch, metrics


def get_missing_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
    )
    return msg


def get_unexpected_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items()
    )
    return msg


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def _group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: List[str]) -> str:
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"


def _named_modules_with_dup(
        model: nn.Module, prefix: str = ""
) -> Iterable[Tuple[str, nn.Module]]:
    """
    The same as `model.named_modules()`, except that it includes
    duplicated modules that have more than one name.
    """
    yield prefix, model
    for name, module in model._modules.items():  # pyre-ignore
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from _named_modules_with_dup(module, submodule_prefix)
        

def bert2vit_ckpt_rename(state_dict,layerCount=8):
    out_Order_dict=OrderedDict({})
    for layer in range(0, layerCount):
        #collect qkv
        bert_q_weight_key="bert.encoder.layer." + str(layer) +".attention.self.query.weight"
        bert_q_bias_key="bert.encoder.layer." + str(layer) +".attention.self.query.bias"
        bert_k_weight_key="bert.encoder.layer." + str(layer) +".attention.self.key.weight"
        bert_k_bias_key="bert.encoder.layer." + str(layer) +".attention.self.key.bias"
        bert_v_weight_key="bert.encoder.layer." + str(layer) +".attention.self.value.weight"
        bert_v_bias_key="bert.encoder.layer." + str(layer) +".attention.self.value.bias"
        pvit_weight_key="blocks." + str(layer) +".attn.qkv.weight"
        pvit_bias_key="blocks." + str(layer) +".attn.qkv.bias"
        mergedQKV_weight= torch.cat((state_dict[bert_q_weight_key],state_dict[bert_k_weight_key],state_dict[bert_v_weight_key]),  0)
        mergedQKV_bias= torch.cat((state_dict[bert_q_bias_key],state_dict[bert_k_bias_key],state_dict[bert_v_bias_key]),  0)     
        out_Order_dict[pvit_weight_key]=mergedQKV_weight
        out_Order_dict[pvit_bias_key]=mergedQKV_bias
    #rename other layers
    for key in state_dict.keys():
        if "attention.output.dense" in key:
            newKey=key.replace("attention.output.dense","attn.proj" )
            newKey=newKey.replace("bert.encoder.layer","blocks" )
            out_Order_dict[newKey]=state_dict[key]
        elif "attention.output.LayerNorm" in key:
            newKey=key.replace("attention.output.LayerNorm","norm1" )
            newKey=newKey.replace("bert.encoder.layer","blocks" )
            out_Order_dict[newKey]=state_dict[key]
        elif "intermediate.dense" in key:
            newKey=key.replace("intermediate.dense","mlp.fc1" )
            newKey=newKey.replace("bert.encoder.layer","blocks" )
            out_Order_dict[newKey]=state_dict[key]
        elif "output.dense" in key and "attention" not in key:
            newKey=key.replace("output.dense","mlp.fc2" )
            newKey=newKey.replace("bert.encoder.layer","blocks" )
            out_Order_dict[newKey]=state_dict[key]
        elif "output.LayerNorm" in key:
            newKey=key.replace("output.LayerNorm","norm2" )
            newKey=newKey.replace("bert.encoder.layer","blocks" )
            out_Order_dict[newKey]=state_dict[key]
    return out_Order_dict
