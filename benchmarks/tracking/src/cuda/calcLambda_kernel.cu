#include "tracking.h"

__global__ void calcLambda_kernel(F2D* cummulative_verticalEdgeSqr, F2D* cummulative_horizontalEdgeSqr, F2D* cummulative_horzVertEdge, int cols, int rows, F2D* lambda)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y;
  
  float cves, ches, chve, tr, det;
  if(tidx < cols && tidy < rows)
  {
    cves = subsref(cummulative_verticalEdgeSqr,tidy,tidx);
    ches = subsref(cummulative_horizontalEdgeSqr,tidy,tidx);
    chve = subsref(cummulative_horzVertEdge,tidy,tidx);
    tr = cves + ches;
    det = cves*ches - chve*chve;
    subsref(lambda,tidy,tidx) = det / (tr+(0.00001));
  }
  
}
