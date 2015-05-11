#ifndef _CSIFT_
#define _CSIFT_

#include "sift.h"

__global__ void imsmooth_kernel(F2D* input, F2D* output, int filterSize, int inTileSize, int outTileSize, float * filter);
__global__ void diffs_kernel(F2D* in1, F2D* in2, F2D* out);
__global__ void halfSize_kernel(F2D * in, F2D * out, int width, int height);

__global__ void imsmoothRow_kernel (F2D* d_Result, F2D* d_Data, int radius, int tileW, float * filter, int dataW, int dataH);
__global__ void imsmoothCol_kernel (F2D* d_Result, F2D* d_Data, int radius, int tileW, float * filter, int dataW, int dataH);


#endif