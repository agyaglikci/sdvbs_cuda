#include "tracking.h"

__global__ void calcSquares_kernel(F2D* verticalEdgeImage, F2D* horizontalEdgeImage, int cols, int rows, F2D* verticalEdgeSq, F2D* horizontalEdgeSq, F2D* horzVertEdge)
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y;

  if(tidx < cols && tidy < rows) {
    float vei = subsref(verticalEdgeImage,tidy,tidx);
    float hei = subsref(horizontalEdgeImage,tidy,tidx);
    subsref(verticalEdgeSq,tidy,tidx) = vei*vei;
    subsref(horzVertEdge,tidy,tidx) = vei*hei;
    subsref(horizontalEdgeSq,tidy,tidx) = hei*hei;
  }
}
