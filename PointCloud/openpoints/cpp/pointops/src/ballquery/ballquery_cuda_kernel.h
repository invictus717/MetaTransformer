#ifndef _BALL_QUERY_GPU_H
#define _BALL_QUERY_GPU_H

#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

int ballquery_cuda(int m, float radius, int nsample,
				   at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
				   at::Tensor offset_tensor, at::Tensor new_offset_tensor,
				   at::Tensor idx_tensor);

void ballquery_launcher(int m, float radius, int nsample,
						const float *xyz, const float *new_xyz,
						const int *offset, const int *new_offset, int *idx);

#endif
