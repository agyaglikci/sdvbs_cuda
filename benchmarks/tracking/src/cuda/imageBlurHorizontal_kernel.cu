#include "tracking.h"

__global__ void imageBlurHorizontal_kernel(I2D* imageIn, F2D* imageOut)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y;

  int rows = imageIn->height;
  int cols = imageIn->width;

  int kernel[] = {1, 4, 6, 4, 1};
  int kernel_size=5;
  int halfkernel = 2;
  int kernelSum = 16;

  if(tidx >= 2 && tidx < cols-2 && tidy >= 2 && tidy < rows-2) {
    float temp=0;
    for(int k=-halfkernel; k<=halfkernel; k++) {
      temp += subsref(imageIn, tidy, tidx+k)*kernel[k+halfkernel];
    }
    subsref(imageOut, tidy, tidx) = temp/kernelSum;
  }
  else if(tidx < cols && tidy < rows) {
    subsref(imageOut, tidy, tidx) = 0;
  }
}
