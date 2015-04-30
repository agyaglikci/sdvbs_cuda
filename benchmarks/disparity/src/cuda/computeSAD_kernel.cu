#include "disparity.h"

__global__ void computeSAD_kernel(I2D* Ileft, I2D* Iright_moved, F2D* SAD)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y;
  if(tidy < Ileft->height && tidx < Ileft->width) {
    int diff = subsref(Ileft,tidy,tidx) - subsref(Iright_moved,tidy,tidx);
    subsref(SAD,tidy,tidx) = diff * diff;
  }
}
