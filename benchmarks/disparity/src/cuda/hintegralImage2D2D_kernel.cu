#include "disparity.h"

__global__ void hintegralImage2D2D_kernel(F2D* SAD, F2D* integralImg)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid < SAD->width)
  {
    int nr = SAD->height;

    subsref(integralImg,0,tid) = subsref(SAD,0,tid);
    for(int i=1; i<nr; i++)
    {
      subsref(integralImg,i,tid) = subsref(integralImg, (i-1), tid) + subsref(SAD,i,tid);
    }
  }
}
