#include "tracking.h"

__global__ void calcSobelHorizontal_dY_kernel(F2D* imageIn, F2D* imageOut)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y;

  int rows = imageIn->height;
  int cols = imageIn->width;

  int kernel_1[] = {1, 2, 1};
  int kernelSum_1=4;
  int kernel_2[] = {1, 0, -1};
  int kernelSum_2=2;
  int halfkernel = 1;

  int startRow = 1;
  int endRow = rows-1;
  int startCol = 1;
  int endCol = cols-1;

  if(tidx >= startCol && tidx < endCol && tidy >= startRow && tidy < endRow) {
    float temp=0;
    for(int k=-halfkernel; k<=halfkernel; k++) {
      temp += subsref(imageIn, tidy, tidx+k)*kernel_1[k+halfkernel];
    }
    subsref(imageOut, tidy, tidx) = temp/kernelSum_2;
  }
  else if(tidx < cols && tidy < rows) {
    subsref(imageOut, tidy, tidx) = 0;
  }
}
