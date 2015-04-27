#include "disparity.h"

__global__ void padarray4_kernel(I2D* inMat, int rows, int cols, int vborder, int hborder, int dir, I2D* paddedArray)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y;

  if(dir == 1)
  {
    if(tidx < cols && tidy < rows) 
    {
      subsref(paddedArray, tidx, tidy) = subsref(inMat, tidx, tidy);
    }
    else if(tidx < cols+hborder && tidy < rows + vborder)
    {
      subsref(paddedArray, tidx, tidy) = 0;
    }
  }
  else 
  {
    if(tidx < hborder && tidy < vborder)
    {
      subsref(paddedArray, tidx, tidy) = 0;
    }
    else if(tidx < cols+hborder && tidy < rows+vborder)
    {
      subsref(paddedArray, tidx, tidy) = subsref(inMat, tidx-hborder, tidy-vborder);
    }
  }
}
