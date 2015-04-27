/********************************
Author: Sravanthi Kota Venkata
********************************/

#ifndef _DISP_
#define _DISP_

extern "C" {
#include "sdvbs_common.h"
}
#include "cudaUtil.h"

void computeSAD(I2D *Ileft, I2D* Iright_moved, F2D* SAD);
I2D* getDisparity(I2D* Ileft, I2D* Iright, int win_sz, int max_shift, int use_gpu, int gpu_transfer);
void finalSAD(F2D* integralImg, int win_sz, F2D* retSAD);
void findDisparity(F2D* retSAD, F2D* minSAD, I2D* retDisp, int level, int nr, int nc);
void integralImage2D2D(F2D* SAD, F2D* integralImg);
void correlateSAD_2D(I2D* Ileft, I2D* Iright, I2D* Iright_moved, int win_sz, int disparity, F2D* SAD, F2D* integralImg, F2D* retSAD);
I2D* padarray2(I2D* inMat, I2D* borderMat);
void padarray4(I2D* inMat, I2D* borderMat, int dir, I2D* paddedArray);
//void BFSGraph(int argc, char** argv);

void cuda_computeSAD(I2D *Ileft, I2D* Iright_moved, F2D* SAD);

void cuda_finalSAD(F2D* integralImg, int win_sz, F2D* retSAD);
void cuda_findDisparity(F2D* retSAD, F2D* minSAD, I2D* retDisp, int level, int nr, int nc);
void cuda_integralImage2D2D(F2D* SAD, F2D* integralImg);
void cuda_correlateSAD_2D(I2D* Ileft, I2D* Iright, I2D* Iright_moved, int win_sz, int disparity, int nr, int nc, F2D* SAD, F2D* integralImg, F2D* retSAD);

/*
//common stuff
F2D* fMallocCudaArray(int nrows, int ncols);
F2D* fMallocCudaArray_copy(F2D* copy);
cudaError_t fCopyToGPU(F2D* host, F2D* device);
cudaError_t fCopyFromGPU(F2D* host, F2D* device);
F2D* fMallocAndCopy(F2D* host_array);
cudaError_t fCopyAndFree(F2D* device_array, F2D* host_array);
I2D* iMallocCudaArray(int nrows, int ncols);
I2D* iMallocCudaArray_copy(I2D* copy);
cudaError_t iCopyToGPU(I2D* host, I2D* device);
cudaError_t iCopyFromGPU(I2D* host, I2D* device);
I2D*  iMallocAndCopy(I2D* host_array);
cudaError_t iCopyAndFree(I2D* device_array, I2D* host_array);
*/

////kernels
//__global__ void padarray4_kernel(I2D* inMat, int rows, int cols, int vborder, int hborder, int dir, I2D* paddedArray);
#endif
