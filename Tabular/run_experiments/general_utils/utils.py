import pickle
from pathlib import Path

import torch
# from pytorch_widedeep.optim import RAdam
from torch.optim import RAdam
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_best_model_args(results_dir: Path, exp_idx: int = None):
    files = sorted(list(results_dir.glob("*")))
    res = []
    for i, f in enumerate(files):
        with open(results_dir / str(f), "rb") as f:  # type: ignore[assignment]
            d = pickle.load(f)  # type: ignore[arg-type]
        res.append((d["early_stopping"].best, d))

    if exp_idx is not None:
        best_params_dict = sorted(res, key=lambda x: x[0])[exp_idx][1]["args"]
    else:
        best_params_dict = sorted(res, key=lambda x: x[0])[0][1]["args"]

    return AttrDict(best_params_dict)


def load_focal_loss_params(results_dir: Path, exp_idx: int):
    files = sorted(list(results_dir.glob("*")))
    res = []
    for i, f in enumerate(files):
        with open(results_dir / str(f), "rb") as f:  # type: ignore[assignment]
            d = pickle.load(f)  # type: ignore[arg-type]
        res.append((d["early_stopping"].best, d))

    fl_best_parms = sorted(res, key=lambda x: x[0])[exp_idx][1]["FLOptimizer"].best
    alpha, gamma = fl_best_parms["alpha"], fl_best_parms["gamma"]

    return alpha, gamma


def steps_up_down(steps_per_epoch, n_epochs, pct_step_up, n_cycles):
    total_steps = steps_per_epoch * n_epochs
    steps_per_cycle = total_steps // n_cycles
    step_size_up = round(steps_per_cycle * pct_step_up)
    step_size_down = int(steps_per_cycle - step_size_up)
    return step_size_up, step_size_down


def set_optimizer(model, args):
    if args.optimizer.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "radam":
        return RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "usedefault":
        return None
    else:
        raise ValueError("Only Adam, AdamW and RAdam are supported for this experiment")


def set_lr_scheduler(optimizer, steps_per_epoch, args):
    if args.lr_scheduler.lower() == "reducelronplateau":
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=args.rop_mode,
            factor=args.rop_factor,
            patience=args.rop_patience,
            threshold=args.rop_threshold,
            threshold_mode=args.rop_threshold_mode,
        )
    elif args.lr_scheduler.lower() == "cycliclr":
        step_size_up, step_size_down = steps_up_down(
            steps_per_epoch, args.n_epochs, args.pct_step_up, args.n_cycles
        )
        lr_scheduler = CyclicLR(
            optimizer,
            base_lr=args.base_lr,
            max_lr=args.max_lr,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            cycle_momentum=args.cycle_momentum,
        )
    elif args.lr_scheduler.lower() == "onecyclelr":
        total_steps = steps_per_epoch * args.n_epochs
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            total_steps=total_steps,
            pct_start=args.pct_step_up,
            div_factor=args.div_factor,
            final_div_factor=args.final_div_factor,
            cycle_momentum=args.cycle_momentum,
        )
    elif args.lr_scheduler.lower() == "noscheduler":
        lr_scheduler = None
    else:
        raise ValueError(
            (
                "Only ReduceLROnPlateau, CyclicLR and OneCycleLR",
                " are supported for this experiment",
            )
        )
    return lr_scheduler
