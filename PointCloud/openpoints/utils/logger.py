import functools
import logging
import os
import os.path as osp
import sys
from termcolor import colored

import time
import shortuuid
import pathlib
import shutil


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger_dist(output=None,
                      distributed_rank=0,
                      *,
                      color=True,
                      name="moco",
                      abbrev_name=None):
    """
    Initialize the detectron2 logger and set its verbosity level to "INFO".
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S")
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
    logging.root = logger  # main logger.
    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")


# ================ experiment folder ==================
def generate_exp_directory(cfg,
                           exp_name=None,
                           expid=None,
                           run_name=None,
                           additional_id=None):
    """Function to create checkpoint folder.
    Args:
        cfg: configuration dict
        cfg.root_dir: the root dir for saving log files.
        exp_name: exp_name or tags for saving and generating the exp_name
        expid: id for the current run
        run_name: the name for the current run. auto generated if None
    Returns:
        the exp_name, jobname, and folders into cfg
    """

    if run_name is None:
        if expid is None:
            expid = time.strftime('%Y%m%d-%H%M%S-') + str(shortuuid.uuid())
            # expid = time.strftime('%Y%m%d-%H%M%S')
        if additional_id is not None:
            expid += '-' + str(additional_id)
        if isinstance(exp_name, list):
            exp_name = '-'.join(exp_name)
        run_name = '-'.join([exp_name, expid])
    cfg.run_name = run_name
    cfg.run_dir = os.path.join(cfg.root_dir, cfg.run_name)
    cfg.exp_dir = cfg.run_dir
    cfg.log_dir = cfg.run_dir
    cfg.ckpt_dir = os.path.join(cfg.run_dir, 'checkpoint')
    cfg.log_path = os.path.join(cfg.run_dir, cfg.run_name + '.log')

    if cfg.get('rank', 0) == 0:
        pathlib.Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)


def resume_exp_directory(cfg, pretrained_path=None):
    """Function to resume the exp folder from the checkpoint folder.
    Args:
        cfg
        pretrained_path: the path to the pretrained model
    Returns:
        the exp_name, jobname, and folders into cfg
    """
    pretrained_path = pretrained_path or cfg.get('pretrained_path', None) or cfg.get('pretrained_path', None)
    if os.path.basename(os.path.dirname(pretrained_path)) == 'checkpoint':
        cfg.run_dir = os.path.dirname(os.path.dirname(cfg.pretrained_path))
        cfg.log_dir = cfg.run_dir
        cfg.run_name = os.path.basename(cfg.run_dir)
        cfg.ckpt_dir = os.path.join(cfg.run_dir, 'checkpoint')
        cfg.code_dir = os.path.join(cfg.run_dir, 'code')
        # we further config the name by datetime
        cfg.log_path = os.path.join(
            cfg.run_dir, cfg.run_name + time.strftime('%Y%m%d-%H%M%S-') +
            str(shortuuid.uuid()) + '.log')
    else:
        expid = time.strftime('%Y%m%d-%H%M%S-') + str(shortuuid.uuid())
        cfg.run_name = '_'.join([os.path.basename(pretrained_path), expid])
        cfg.run_dir = os.path.join(cfg.root_dir, cfg.run_name)
        cfg.log_dir = cfg.run_dir
        cfg.ckpt_dir = os.path.join(cfg.run_dir, 'checkpoint')
        cfg.code_dir = os.path.join(cfg.run_dir, 'code')
        cfg.log_path = os.path.join(cfg.run_dir, cfg.run_name + '.log')
        
    if cfg.get('rank', 0) == 0:
        os.makedirs(cfg.run_dir, exist_ok=True)
    cfg.wandb.tags = ['resume']