/*
batch version of ball query, modified from the original implementation of official PointNet++ codes.
Written by PointNeXt team 
All Rights Reserved 2022.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../cuda_utils.h"
#include "ballquery_cuda_kernel.h"

__device__ int ballquery_bt_idx(int idx, const int *offset)
{
    int i = 0;
    while (1)
    {
        if (idx < offset[i])
            break;
        else
            i++;
    }
    return i;
}

__global__ void ballquery_cuda_kernel(int m, float radius, int nsample,
                                      const float *__restrict__ xyz, const float *__restrict__ new_xyz,
                                      const int *__restrict__ offset, const int *__restrict__ new_offset,
                                      int *__restrict__ idx)
{
    // input: xyz (n, 3) new_xyz (m, 3) offset new_offset
    // output: idx (m, nsample)

    // only 1-d idx is needed. idx in point dimension
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m)
        return;

    new_xyz += pt_idx * 3;
    idx += pt_idx * nsample;

    int bt_idx = ballquery_bt_idx(pt_idx, new_offset);
    int start;
    if (bt_idx == 0)
        start = 0;
    else
        start = offset[bt_idx - 1];
    int end = offset[bt_idx];

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    // for each m, this for loop is done in cuda luncher
    int cnt = 0;
    for (int k = start; k < end; ++k)
    {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2)
        {
            if (cnt == 0) // init as k
            {
                for (int l = 0; l < nsample; ++l)
                {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample)
                break;
        }
    }
}

void ballquery_launcher(int m, float radius, int nsample,
                        const float *xyz, const float *new_xyz,
                        const int *offset, const int *new_offset,
                        int *idx)
{
    // input: xyz (n, 3) new_xyz (m, 3) offset new_offset
    // output: idx (m, nsample)
    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK)); // only 1 dimension. in point dimension, perform multi-threads
    dim3 threads(THREADS_PER_BLOCK);

    ballquery_cuda_kernel<<<blocks, threads, 0>>>(m, radius, nsample, xyz, new_xyz, offset, new_offset, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
