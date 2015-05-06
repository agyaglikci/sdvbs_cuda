/********************************
Author: Sravanthi Kota Venkata
********************************/

#ifndef _SIFT_
#define _SIFT_

extern "C" {
#include "sdvbs_common.h"
}
#include "cudaUtil.h"

#include <assert.h>

#define GREATER(a,b) ((a) > (b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

F2D* sift(F2D* I, int use_gpu, int gpu_transfer);
F2D* halveSize(F2D* I);
F2D** gaussianss(F2D* I, float sigman, int O, int S, int omin, int smin, int smax, float sigma0, int use_gpu, int gpu_transfer);
F2D** diffss(F2D** ss, int O, int intervals);
F2D* doubleSize(F2D* I);
void imsmooth(F2D* I_pt, float dsigma, F2D* out, int gpu_transfer);
F2D* siftlocalmax(F2D* in, float thresh, int intervals, int M, int N);
F2D* filterBoundaryPoints(int M, int N, F2D* oframes);
F2D* siftrefinemx(F2D* oframes, F2D* dogss, int smin, float thresh, int rin, int M, int N, int intervals);



void cuda_imsmooth(F2D* array, float dsigma, F2D* out);
F2D** cuda_gaussianss(F2D* array, float sigman, int O, int S, int omin, int smin, int smax, float sigma0);

#endif
