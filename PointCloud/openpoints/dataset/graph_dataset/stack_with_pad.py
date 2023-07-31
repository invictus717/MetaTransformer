import numpy as np
import numba as nb


@nb.njit
def stack_with_pad_4d(inputs):
    num_elem = len(inputs)
    ms_0, ms_1, ms_2, ms_3 = inputs[0].shape
    
    for i in range(1,num_elem):
        is_0, is_1, is_2, is_3 = inputs[i].shape
        ms_0 = max(is_0, ms_0)
        ms_1 = max(is_1, ms_1)
        ms_2 = max(is_2, ms_2)
        ms_3 = max(is_3, ms_3)
    
    stacked_shape = (num_elem,ms_0,ms_1,ms_2,ms_3)
    stacked = np.zeros(stacked_shape, dtype=inputs[0].dtype)
    
    for i, elem in enumerate(inputs):
        stacked[i][:elem.shape[0],:elem.shape[1],:elem.shape[2],:elem.shape[3]] = elem
    return stacked

@nb.njit
def stack_with_pad_3d(inputs):
    num_elem = len(inputs)
    ms_0, ms_1, ms_2 = inputs[0].shape
    
    for i in range(1,num_elem):
        is_0, is_1, is_2 = inputs[i].shape
        ms_0 = max(is_0, ms_0)
        ms_1 = max(is_1, ms_1)
        ms_2 = max(is_2, ms_2)
    
    stacked_shape = (num_elem,ms_0,ms_1,ms_2)
    stacked = np.zeros(stacked_shape, dtype=inputs[0].dtype)
    
    for i, elem in enumerate(inputs):
        stacked[i][:elem.shape[0],:elem.shape[1],:elem.shape[2]] = elem
    return stacked

@nb.njit
def stack_with_pad_2d(inputs):
    num_elem = len(inputs)
    ms_0, ms_1 = inputs[0].shape
    
    for i in range(1,num_elem):
        is_0, is_1 = inputs[i].shape
        ms_0 = max(is_0, ms_0)
        ms_1 = max(is_1, ms_1)
    
    stacked_shape = (num_elem,ms_0,ms_1)
    stacked = np.zeros(stacked_shape, dtype=inputs[0].dtype)
    
    for i, elem in enumerate(inputs):
        stacked[i][:elem.shape[0],:elem.shape[1]] = elem
    return stacked

@nb.njit
def stack_with_pad_1d(inputs):
    num_elem = len(inputs)
    ms_0 = inputs[0].shape[0]
    
    for i in range(1,num_elem):
        is_0 = inputs[i].shape[0]
        ms_0 = max(is_0, ms_0)
    
    stacked_shape = (num_elem,ms_0)
    stacked = np.zeros(stacked_shape, dtype=inputs[0].dtype)
    
    for i, elem in enumerate(inputs):
        stacked[i][:elem.shape[0]] = elem
    return stacked


def stack_with_pad(inputs):
    shape_rank = np.ndim(inputs[0])
    if shape_rank == 0:
        return np.stack(inputs)
    if shape_rank == 1:
        return stack_with_pad_1d(inputs)
    elif shape_rank == 2:
        return stack_with_pad_2d(inputs)
    elif shape_rank == 3:
        return stack_with_pad_3d(inputs)
    elif shape_rank == 4:
        return stack_with_pad_4d(inputs)
    else:
        raise ValueError('Only support up to 4D tensor')


