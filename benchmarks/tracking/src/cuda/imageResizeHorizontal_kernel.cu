#include "tracking.h"

__global__ void imageResizeHorizontal_kernel(F2D* imageIn, F2D* imageOut)
{
  int tidx_out = blockIdx.x*blockDim.x + threadIdx.x;
  int tidx_in = 2*(tidx_out)+2;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y;

  int kernel[] = {1, 4, 6, 4, 1};
  int halfkernel = 2;
  int kernelSum = 16;

  int rows = imageIn->height;
  int in_cols = imageIn->width;
  int out_cols = imageOut->width;

  if( tidx_in < in_cols-2 && tidy >=2 && tidy < rows-2)
  {
    float temp=0;
    for(int k=-halfkernel; k<=halfkernel; k++) {
      temp += subsref(imageIn, tidy, tidx_in+k)*kernel[k+halfkernel];
    }
    subsref(imageOut, tidy, tidx_out) = temp/kernelSum;
  }
  else if(tidx_out < out_cols && tidy < rows) {
    subsref(imageOut, tidy, tidx_out) = 0;
  }
}

