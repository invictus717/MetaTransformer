/**********************************
 * Original Author: Haoqiang Fan
 * Modified by: Kaichun Mo
 *********************************/

#ifndef _EMD_KERNEL
#define _EMD_KERNEL

#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>  // at::cuda::getApplyGrid
// #include <THC/THC.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/********************************
* Forward kernel for approxmatch
*********************************/

template<typename scalar_t>
__global__ void approxmatch(int b,int n,int m,const scalar_t * __restrict__ xyz1,const scalar_t * __restrict__ xyz2,scalar_t * __restrict__ match,scalar_t * temp){
	scalar_t * remainL=temp+blockIdx.x*(n+m)*2, * remainR=temp+blockIdx.x*(n+m)*2+n,*ratioL=temp+blockIdx.x*(n+m)*2+n+m,*ratioR=temp+blockIdx.x*(n+m)*2+n+m+n;
	scalar_t multiL,multiR;
	if (n>=m){
		multiL=1;
		multiR=n/m;
	}else{
		multiL=m/n;
		multiR=1;
	}
	const int Block=1024;
	__shared__ scalar_t buf[Block*4];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x;j<n*m;j+=blockDim.x)
			match[i*n*m+j]=0;
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			remainL[j]=multiL;
		for (int j=threadIdx.x;j<m;j+=blockDim.x)
			remainR[j]=multiR;
		__syncthreads();
		for (int j=7;j>=-2;j--){
			scalar_t level=-powf(4.0f,j);
			if (j==-2){
				level=0;
			}
			for (int k0=0;k0<n;k0+=blockDim.x){
				int k=k0+threadIdx.x;
				scalar_t x1=0,y1=0,z1=0;
				if (k<n){
					x1=xyz1[i*n*3+k*3+0];
					y1=xyz1[i*n*3+k*3+1];
					z1=xyz1[i*n*3+k*3+2];
				}
				scalar_t suml=1e-9f;
				for (int l0=0;l0<m;l0+=Block){
					int lend=min(m,l0+Block)-l0;
					for (int l=threadIdx.x;l<lend;l+=blockDim.x){
						scalar_t x2=xyz2[i*m*3+l0*3+l*3+0];
						scalar_t y2=xyz2[i*m*3+l0*3+l*3+1];
						scalar_t z2=xyz2[i*m*3+l0*3+l*3+2];
						buf[l*4+0]=x2;
						buf[l*4+1]=y2;
						buf[l*4+2]=z2;
						buf[l*4+3]=remainR[l0+l];
					}
					__syncthreads();
					for (int l=0;l<lend;l++){
						scalar_t x2=buf[l*4+0];
						scalar_t y2=buf[l*4+1];
						scalar_t z2=buf[l*4+2];
						scalar_t d=level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
						scalar_t w=__expf(d)*buf[l*4+3];
						suml+=w;
					}
					__syncthreads();
				}
				if (k<n)
					ratioL[k]=remainL[k]/suml;
			}
			__syncthreads();
			for (int l0=0;l0<m;l0+=blockDim.x){
				int l=l0+threadIdx.x;
				scalar_t x2=0,y2=0,z2=0;
				if (l<m){
					x2=xyz2[i*m*3+l*3+0];
					y2=xyz2[i*m*3+l*3+1];
					z2=xyz2[i*m*3+l*3+2];
				}
				scalar_t sumr=0;
				for (int k0=0;k0<n;k0+=Block){
					int kend=min(n,k0+Block)-k0;
					for (int k=threadIdx.x;k<kend;k+=blockDim.x){
						buf[k*4+0]=xyz1[i*n*3+k0*3+k*3+0];
						buf[k*4+1]=xyz1[i*n*3+k0*3+k*3+1];
						buf[k*4+2]=xyz1[i*n*3+k0*3+k*3+2];
						buf[k*4+3]=ratioL[k0+k];
					}
					__syncthreads();
					for (int k=0;k<kend;k++){
						scalar_t x1=buf[k*4+0];
						scalar_t y1=buf[k*4+1];
						scalar_t z1=buf[k*4+2];
						scalar_t w=__expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*buf[k*4+3];
						sumr+=w;
					}
					__syncthreads();
				}
				if (l<m){
					sumr*=remainR[l];
					scalar_t consumption=fminf(remainR[l]/(sumr+1e-9f),1.0f);
					ratioR[l]=consumption*remainR[l];
					remainR[l]=fmaxf(0.0f,remainR[l]-sumr);
				}
			}
			__syncthreads();
			for (int k0=0;k0<n;k0+=blockDim.x){
				int k=k0+threadIdx.x;
				scalar_t x1=0,y1=0,z1=0;
				if (k<n){
					x1=xyz1[i*n*3+k*3+0];
					y1=xyz1[i*n*3+k*3+1];
					z1=xyz1[i*n*3+k*3+2];
				}
				scalar_t suml=0;
				for (int l0=0;l0<m;l0+=Block){
					int lend=min(m,l0+Block)-l0;
					for (int l=threadIdx.x;l<lend;l+=blockDim.x){
						buf[l*4+0]=xyz2[i*m*3+l0*3+l*3+0];
						buf[l*4+1]=xyz2[i*m*3+l0*3+l*3+1];
						buf[l*4+2]=xyz2[i*m*3+l0*3+l*3+2];
						buf[l*4+3]=ratioR[l0+l];
					}
					__syncthreads();
					scalar_t rl=ratioL[k];
					if (k<n){
						for (int l=0;l<lend;l++){
							scalar_t x2=buf[l*4+0];
							scalar_t y2=buf[l*4+1];
							scalar_t z2=buf[l*4+2];
							scalar_t w=__expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*rl*buf[l*4+3];
							match[i*n*m+(l0+l)*n+k]+=w;
							suml+=w;
						}
					}
					__syncthreads();
				}
				if (k<n)
					remainL[k]=fmaxf(0.0f,remainL[k]-suml);
			}
			__syncthreads();
		}
	}
}

//void approxmatchLauncher(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,scalar_t * match,scalar_t * temp){
//	approxmatch<<<32,512>>>(b,n,m,xyz1,xyz2,match,temp);
//}

/* ApproxMatch forward interface
Input:
  xyz1: (B, N1, 3)  # dataset_points
  xyz2: (B, N2, 3)  # query_points
Output:
  match: (B, N2, N1)
*/
at::Tensor ApproxMatchForward(
    const at::Tensor xyz1,
    const at::Tensor xyz2){
  const auto b = xyz1.size(0);
  const auto n = xyz1.size(1);
  const auto m = xyz2.size(1);

  CHECK_EQ(xyz2.size(0), b);
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);

  auto match = at::zeros({b, m, n}, xyz1.type());
  auto temp = at::zeros({b, (n+m)*2}, xyz1.type());

  AT_DISPATCH_FLOATING_TYPES(xyz1.scalar_type(), "ApproxMatchForward", ([&] {
        approxmatch<scalar_t><<<32,512>>>(b, n, m, xyz1.data<scalar_t>(), xyz2.data<scalar_t>(), match.data<scalar_t>(), temp.data<scalar_t>());
  }));
  //THCudaCheck(cudaGetLastError());

  return match;
}


/********************************
* Forward kernel for matchcost
*********************************/

template<typename scalar_t>
__global__ void matchcost(int b,int n,int m,const scalar_t * __restrict__ xyz1,const scalar_t * __restrict__ xyz2,const scalar_t * __restrict__ match,scalar_t * __restrict__ out){
	__shared__ scalar_t allsum[512];
	const int Block=1024;
	__shared__ scalar_t buf[Block*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		scalar_t subsum=0;
		for (int k0=0;k0<n;k0+=blockDim.x){
			int k=k0+threadIdx.x;
			scalar_t x1=0,y1=0,z1=0;
			if (k<n){
				x1=xyz1[i*n*3+k*3+0];
				y1=xyz1[i*n*3+k*3+1];
				z1=xyz1[i*n*3+k*3+2];
			}
			for (int l0=0;l0<m;l0+=Block){
				int lend=min(m,l0+Block)-l0;
				for (int l=threadIdx.x;l<lend*3;l+=blockDim.x)
					buf[l]=xyz2[i*m*3+l0*3+l];
				__syncthreads();
				if (k<n){
					for (int l=0;l<lend;l++){
						scalar_t x2=buf[l*3+0];
						scalar_t y2=buf[l*3+1];
						scalar_t z2=buf[l*3+2];
						scalar_t d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
						subsum+=d*match[i*n*m+(l0+l)*n+k];
					}
				}
				__syncthreads();
			}
		}
		allsum[threadIdx.x]=subsum;
		for (int j=1;j<blockDim.x;j<<=1){
			__syncthreads();
			if ((threadIdx.x&j)==0 && threadIdx.x+j<blockDim.x){
				allsum[threadIdx.x]+=allsum[threadIdx.x+j];
			}
		}
		if (threadIdx.x==0)
			out[i]=allsum[0];
		__syncthreads();
	}
}

//void matchcostLauncher(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,const scalar_t * match,scalar_t * out){
//	matchcost<<<32,512>>>(b,n,m,xyz1,xyz2,match,out);
//}

/* MatchCost forward interface
Input:
  xyz1: (B, N1, 3)  # dataset_points
  xyz2: (B, N2, 3)  # query_points
  match: (B, N2, N1)
Output:
  cost: (B)
*/
at::Tensor MatchCostForward(
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const at::Tensor match){
  const auto b = xyz1.size(0);
  const auto n = xyz1.size(1);
  const auto m = xyz2.size(1);

  CHECK_EQ(xyz2.size(0), b);
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);

  auto cost = at::zeros({b}, xyz1.type());

  AT_DISPATCH_FLOATING_TYPES(xyz1.scalar_type(), "MatchCostForward", ([&] {
        matchcost<scalar_t><<<32,512>>>(b, n, m, xyz1.data<scalar_t>(), xyz2.data<scalar_t>(), match.data<scalar_t>(), cost.data<scalar_t>());
  }));
  //THCudaCheck(cudaGetLastError());

  return cost;
}


/********************************
* matchcostgrad2 kernel
*********************************/

template<typename scalar_t>
__global__ void matchcostgrad2(int b,int n,int m,const scalar_t * __restrict__ grad_cost,const scalar_t * __restrict__ xyz1,const scalar_t * __restrict__ xyz2,const scalar_t * __restrict__ match,scalar_t * __restrict__ grad2){
	__shared__ scalar_t sum_grad[256*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		int kbeg=m*blockIdx.y/gridDim.y;
		int kend=m*(blockIdx.y+1)/gridDim.y;
		for (int k=kbeg;k<kend;k++){
			scalar_t x2=xyz2[(i*m+k)*3+0];
			scalar_t y2=xyz2[(i*m+k)*3+1];
			scalar_t z2=xyz2[(i*m+k)*3+2];
			scalar_t subsumx=0,subsumy=0,subsumz=0;
			for (int j=threadIdx.x;j<n;j+=blockDim.x){
				scalar_t x1=x2-xyz1[(i*n+j)*3+0];
				scalar_t y1=y2-xyz1[(i*n+j)*3+1];
				scalar_t z1=z2-xyz1[(i*n+j)*3+2];
				scalar_t d=match[i*n*m+k*n+j]*2;
				subsumx+=x1*d;
				subsumy+=y1*d;
				subsumz+=z1*d;
			}
			sum_grad[threadIdx.x*3+0]=subsumx;
			sum_grad[threadIdx.x*3+1]=subsumy;
			sum_grad[threadIdx.x*3+2]=subsumz;
			for (int j=1;j<blockDim.x;j<<=1){
				__syncthreads();
				int j1=threadIdx.x;
				int j2=threadIdx.x+j;
				if ((j1&j)==0 && j2<blockDim.x){
					sum_grad[j1*3+0]+=sum_grad[j2*3+0];
					sum_grad[j1*3+1]+=sum_grad[j2*3+1];
					sum_grad[j1*3+2]+=sum_grad[j2*3+2];
				}
			}
			if (threadIdx.x==0){
				grad2[(i*m+k)*3+0]=sum_grad[0]*grad_cost[i];
				grad2[(i*m+k)*3+1]=sum_grad[1]*grad_cost[i];
				grad2[(i*m+k)*3+2]=sum_grad[2]*grad_cost[i];
			}
			__syncthreads();
		}
	}
}

/********************************
* matchcostgrad1 kernel
*********************************/

template<typename scalar_t>
__global__ void matchcostgrad1(int b,int n,int m,const scalar_t * __restrict__ grad_cost,const scalar_t * __restrict__ xyz1,const scalar_t * __restrict__ xyz2,const scalar_t * __restrict__ match,scalar_t * __restrict__ grad1){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int l=threadIdx.x;l<n;l+=blockDim.x){
			scalar_t x1=xyz1[i*n*3+l*3+0];
			scalar_t y1=xyz1[i*n*3+l*3+1];
			scalar_t z1=xyz1[i*n*3+l*3+2];
			scalar_t dx=0,dy=0,dz=0;
			for (int k=0;k<m;k++){
				scalar_t x2=xyz2[i*m*3+k*3+0];
				scalar_t y2=xyz2[i*m*3+k*3+1];
				scalar_t z2=xyz2[i*m*3+k*3+2];
				scalar_t d=match[i*n*m+k*n+l]*2;
				dx+=(x1-x2)*d;
				dy+=(y1-y2)*d;
				dz+=(z1-z2)*d;
			}
			grad1[i*n*3+l*3+0]=dx*grad_cost[i];
			grad1[i*n*3+l*3+1]=dy*grad_cost[i];
			grad1[i*n*3+l*3+2]=dz*grad_cost[i];
		}
	}
}

//void matchcostgradLauncher(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,const scalar_t * match,scalar_t * grad1,scalar_t * grad2){
//	matchcostgrad1<<<32,512>>>(b,n,m,xyz1,xyz2,match,grad1);
//	matchcostgrad2<<<dim3(32,32),256>>>(b,n,m,xyz1,xyz2,match,grad2);
//}


/* MatchCost backward interface
Input:
  grad_cost: (B)    # gradients on cost
  xyz1: (B, N1, 3)  # dataset_points
  xyz2: (B, N2, 3)  # query_points
  match: (B, N2, N1)
Output:
  grad1: (B, N1, 3)
  grad2: (B, N2, 3)
*/
std::vector<at::Tensor> MatchCostBackward(
    const at::Tensor grad_cost,
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const at::Tensor match){
  const auto b = xyz1.size(0);
  const auto n = xyz1.size(1);
  const auto m = xyz2.size(1);

  CHECK_EQ(xyz2.size(0), b);
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);

  auto grad1 = at::zeros({b, n, 3}, xyz1.type());
  auto grad2 = at::zeros({b, m, 3}, xyz1.type());

  AT_DISPATCH_FLOATING_TYPES(xyz1.scalar_type(), "MatchCostBackward", ([&] {
        matchcostgrad1<scalar_t><<<32,512>>>(b, n, m, grad_cost.data<scalar_t>(), xyz1.data<scalar_t>(), xyz2.data<scalar_t>(), match.data<scalar_t>(), grad1.data<scalar_t>());
        matchcostgrad2<scalar_t><<<dim3(32,32),256>>>(b, n, m, grad_cost.data<scalar_t>(), xyz1.data<scalar_t>(), xyz2.data<scalar_t>(), match.data<scalar_t>(), grad2.data<scalar_t>());
  }));
  //THCudaCheck(cudaGetLastError());

  return std::vector<at::Tensor>({grad1, grad2});
}

#endif
