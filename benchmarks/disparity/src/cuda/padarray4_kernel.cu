#include "disparity.h"

__global__ void padarray4_kernel(I2D* inMat, int vborder, int hborder, int dir, I2D* paddedArray)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y;

  int rows = inMat->height;
  int cols = inMat->width;

  if(dir == 1)
  {
    if(tidx < cols || tidy < rows) 
    {
      subsref(paddedArray, tidy, tidx) = subsref(inMat, tidy, tidx);
    }
    else if(tidx < cols/*+hborder*/ && tidy < rows/* + vborder*/)
    {
      subsref(paddedArray, tidy, tidx) = 0;
    }
  }
  else 
  {
    if(tidx < hborder || tidy < vborder)
    {
      subsref(paddedArray, tidy, tidx) = 0;
    }
    else if(tidx < cols/*+hborder*/ && tidy < rows/*+vborder*/)
    {
      subsref(paddedArray, tidy, tidx) = subsref(inMat, tidy-vborder, tidx-hborder);
    }
  }
}
