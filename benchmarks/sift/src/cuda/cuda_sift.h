#ifndef _CSIFT_
#define _CSIFT_

#include "sift.h"

__global__ void imsmooth_kernel(F2D* input, F2D* output, int filterSize, int inTileSize, int outTileSize, float * filter);

#endif