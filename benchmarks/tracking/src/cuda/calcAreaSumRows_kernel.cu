#include "tracking.h"

__global__ void calcAreaSumRows_kernel(F2D* imageIn, F2D* imageOut, int winSize)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  //int tidy = blockIdx.y*blockDim.y + threadIdx.y;
  
  int rows = imageIn->height;
  int cols = imageIn->width;
  
  int halfWindow = ((winSize+1)/2);
  int remainder = winSize - halfWindow;
  /*
  
  if(tidx < cols && tidy < rows) {
    float windowSum=0;
    int winstart = tidx - halfWindow;
    int winstop = tidx + halfWindow-1;
    winstart = winstart >= 0 ? winstart : 0;
    winstop = winstop < cols ? winstop : cols-1;
    for(int k=winstart; k<=winstop; k++) {
      windowSum += subsref(imageIn, tidy, k);
    }
    subsref(imageOut, tidy, tidx) = windowSum;
  }
  */

  if(tidx < rows) {
    float a1sum=0;
    for(int i=0; i<remainder; i++) {
      a1sum += subsref(imageIn,tidx,i);
    }
    for(int i=0; i<cols; i++)
    {
      subsref(imageOut,tidx,i) = a1sum;
      float to_add = (i+remainder >= cols ? 0 : subsref(imageIn,tidx,i+remainder));
      float to_sub = (i-halfWindow < 0 ? 0 : subsref(imageIn,tidx,i-halfWindow));
      a1sum += to_add - to_sub;
    }
  }
}
