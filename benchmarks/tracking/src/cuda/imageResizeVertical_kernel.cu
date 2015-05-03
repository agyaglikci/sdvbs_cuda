#include "tracking.h"

__global__ void imageResizeVertical_kernel(F2D* imageIn, F2D* imageOut)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int tidy_out = blockIdx.y*blockDim.y + threadIdx.y;
  int tidy_in = 2*tidy_out+2;

  int kernel[] = {1, 4, 6, 4, 1};
  int halfkernel = 2;
  int kernelSum = 16;

  int cols = imageIn->width;
  int in_rows = imageIn->height;
  int out_rows = imageOut->height;

  if( tidx < cols && tidy_in < in_rows-2) 
  {
    float temp=0;
    for(int k=-halfkernel; k<=halfkernel; k++) {
      temp += subsref(imageIn, tidy_in+k, tidx)*kernel[k+halfkernel];
    }
    subsref(imageOut, tidy_out, tidx) = temp/kernelSum;
  }
  else if(tidx < cols && tidy_out < out_rows) {
    subsref(imageOut, tidy_out, tidx) = 0;
  }
}

