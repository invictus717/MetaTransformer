import numpy as np
import random
import torch 


def set_random_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if deterministic:
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False