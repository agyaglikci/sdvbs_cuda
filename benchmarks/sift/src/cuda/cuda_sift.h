#ifndef _CSIFT_
#define _CSIFT_

#include "sift.h"

__global__ void imsmooth_kernel(F2D* input, F2D* output, int filterSize, int inTileSize, int outTileSize, float * filter);
__global__ void diffs_kernel(F2D* in1, F2D* in2, F2D* out);

#endif