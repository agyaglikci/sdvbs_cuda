#include "disparity.h"

__global__ void findDisparity_kernel(F2D* retSAD, F2D* minSAD, I2D* retDisp, int level, int nr, int nc)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y;

  if(tidx < nc && tidy < nr)
  {
    int a = subsref(retSAD, tidy, tidx);
    int b = subsref(minSAD, tidy, tidx);
    if(a<b)
    {
      subsref(minSAD,tidy,tidx) = a;
      subsref(retDisp,tidy,tidx) = level; 
    }
  }
}
