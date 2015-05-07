#include "tracking.h"

__global__ void calcAreaSumCols_kernel(F2D* imageIn, F2D* imageOut, int winSize)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  //int tidy = blockIdx.y*blockDim.y + threadIdx.y;
  
  int rows = imageIn->height;
  int cols = imageIn->width;
  
  int halfWindow = (winSize+1)/2;
  int remainder = winSize - halfWindow;
  /*
  if(tidx < cols && tidy < rows) {
    float windowSum=0;
    int winstart = tidy - halfWindow;
    int winstop = tidy + halfWindow -1;
    winstart = winstart > 0 ? winstart : 0;
    winstop = winstop < rows ? winstop : rows-1;
    for(int k=winstart; k<=winstop; k++) {
      windowSum += subsref(imageIn, k, tidx);
    }
    subsref(imageOut, tidy, tidx) = windowSum;
  }
  */

  if(tidx < cols) {
    float a1sum=0;
    for(int i=0; i<winSize-halfWindow; i++) {
      a1sum += subsref(imageIn,i,tidx);
    }
    for(int i=0; i<rows; i++)
    {
      subsref(imageOut,i,tidx) = a1sum;
      float to_add = (i+remainder >= rows ? 0 : subsref(imageIn,i+remainder,tidx));
      float to_sub = (i-halfWindow < 0 ? 0 : subsref(imageIn,i-halfWindow,tidx));
      a1sum += to_add - to_sub;
    }
  }

}
