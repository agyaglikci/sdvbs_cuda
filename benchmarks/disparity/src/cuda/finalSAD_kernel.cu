#include "disparity.h"

__global__ void finalSAD_kernel(F2D* integralImg, int win_sz, F2D* retSAD)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y;

  int endR = integralImg->height;
  int endC = integralImg->width;

  if( tidy<endR-win_sz && tidx<endC-win_sz) {
    subsref(retSAD,tidy,tidx) = subsref(integralImg,(tidy+win_sz),(tidx+win_sz)) 
          + subsref(integralImg,(tidy+1),(tidx+1)) 
          - subsref(integralImg,(tidy+1),(tidx+win_sz))
          - subsref(integralImg,(tidy+win_sz),(tidx+1));
  }
}
