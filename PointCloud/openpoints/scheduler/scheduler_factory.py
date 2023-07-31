""" Scheduler Factory
Borrowed from Ross Wightman (https://www.github.com/timm)
"""
from .cosine_lr import CosineLRScheduler
from .multistep_lr import MultiStepLRScheduler
from .plateau_lr import PlateauLRScheduler
from .poly_lr import PolyLRScheduler
from .step_lr import StepLRScheduler
from .tanh_lr import TanhLRScheduler


def build_scheduler_from_cfg(args, optimizer, return_epochs=False):
    num_epochs = args.epochs
    warmup_epochs = getattr(args, 'warmup_epochs', 0)
    warmup_lr = getattr(args, 'warmup_lr', 1.0e-6)  # linear warmup
    min_lr = args.min_lr if getattr(args, 'min_lr', False) else args.lr/1000.
    cooldown_epochs = getattr(args, 'cooldown_epochs', 0) 
    final_decay_rate = getattr(args, 'final_decay_rate', 0.01)
    decay_rate = getattr(args, 'decay_rate', None) or final_decay_rate**(1/num_epochs)
    decay_epochs = getattr(args, 'decay_epochs', 1)
    t_max = getattr(args, 't_max', num_epochs)
    if getattr(args, 'lr_noise', None) is not None:
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(args, 'lr_noise_pct', 0.67),
        noise_std=getattr(args, 'lr_noise_std', 1.),
        noise_seed=getattr(args, 'seed', 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(args, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(args, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(args, 'lr_cycle_limit', 1),
    )

    lr_scheduler = None
    if args.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_max,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            k_decay=getattr(args, 'lr_k_decay', 1.0),
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + cooldown_epochs
    elif args.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            t_in_epochs=True,
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + cooldown_epochs
    elif args.sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_epochs,
            decay_rate=decay_rate,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            **noise_args,
        )
    elif args.sched == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=decay_epochs,
            decay_rate=decay_rate,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            **noise_args,
        )
    elif args.sched == 'plateau':
        mode = 'min' if 'loss' in getattr(args, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=args.decay_rate,
            patience_t=args.patience_epochs,
            lr_min=min_lr,
            mode=mode,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            cooldown_t=0,
            **noise_args,
        )
    elif args.sched == 'poly':
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=args.decay_rate,  # overloading 'decay_rate' as polynomial power
            t_initial=num_epochs,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            k_decay=getattr(args, 'lr_k_decay', 1.0),
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + cooldown_epochs

    if return_epochs:
        return lr_scheduler, num_epochs
    else:
        return lr_scheduler