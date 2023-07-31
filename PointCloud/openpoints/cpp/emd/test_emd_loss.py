import torch
import numpy as np
import time
from emd import earth_mover_distance

# gt
p1 = torch.from_numpy(np.array([[[1.7, -0.1, 0.1], [0.1, 1.2, 0.3]]], dtype=np.float32)).cuda()
p1 = p1.repeat(3, 1, 1)
p2 = torch.from_numpy(np.array([[[0.3, 1.8, 0.2], [1.2, -0.2, 0.3]]], dtype=np.float32)).cuda()
p2 = p2.repeat(3, 1, 1)
print(p1)
print(p2)
print(p1.shape)
p1.requires_grad = True
p2.requires_grad = True

gt_dist = (((p1[0, 0] - p2[0, 1])**2).sum() + ((p1[0, 1] - p2[0, 0])**2).sum()) / 2 +  \
         (((p1[1, 0] - p2[1, 1])**2).sum() + ((p1[1, 1] - p2[1, 0])**2).sum()) * 2 + \
         (((p1[2, 0] - p2[2, 1])**2).sum() + ((p1[2, 1] - p2[2, 0])**2).sum()) / 3
print('gt_dist: ', gt_dist)

gt_dist.backward()
print(p1.grad)
print(p2.grad)

# emd
p1 = torch.from_numpy(np.array([[[1.7, -0.1, 0.1], [0.1, 1.2, 0.3]]], dtype=np.float32)).cuda()
p1 = p1.repeat(3, 1, 1)
p2 = torch.from_numpy(np.array([[[0.3, 1.8, 0.2], [1.2, -0.2, 0.3]]], dtype=np.float32)).cuda()
p2 = p2.repeat(3, 1, 1)
print(p1)
print(p2)
p1.requires_grad = True
p2.requires_grad = True

d = earth_mover_distance(p1, p2, transpose=False)
print(d)

loss = d[0] / 2 + d[1] * 2 + d[2] / 3
print(loss)

loss.backward()
print(p1.grad)
print(p2.grad)

