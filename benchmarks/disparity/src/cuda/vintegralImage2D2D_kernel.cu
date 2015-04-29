#include "disparity.h"

__global__ void vintegralImage2D2D_kernel(F2D* SAD, F2D* integralImg)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < SAD->height)
  {
    int nc = SAD->width;

    for(int i=1; i<nc; i++)
    {
      subsref(integralImg,tid,i) = subsref(integralImg,tid,(i-1)) + subsref(integralImg,tid,i);
    }
  }
}
