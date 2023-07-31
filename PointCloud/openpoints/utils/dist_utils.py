import os
import torch
from torch import distributed as dist


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    print(f'init distributed in rank {torch.distributed.get_rank()}')


def get_dist_info(cfg):
    mp = False
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        cfg.launcher = 'pytorch'
    else:
        # only supports 1 node for now
        rank = cfg.local_rank
        world_size = torch.cuda.device_count()
        mp = cfg.launcher in ['mp', 'multiprocessing'] and world_size > 1
    distributed = world_size > 1
    print(f'launch {cfg.launcher} with {world_size} GPUs, current rank: {rank}')
    return rank, world_size, distributed, mp


def reduce_tensor(tensor):
    '''
        for acc kind, get the mean in each gpu
    '''
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def gather_tensor(tensor):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port