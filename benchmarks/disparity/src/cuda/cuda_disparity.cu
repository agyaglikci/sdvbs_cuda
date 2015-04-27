#include <stdio.h>
#include <stdlib.h>
#include "disparity.h"


//void cuda_computeSAD(I2D *Ileft, I2D* Iright_moved, F2D* SAD);
//
//void cuda_finalSAD(F2D* integralImg, int win_sz, F2D* retSAD);
//void cuda_findDisparity(F2D* retSAD, F2D* minSAD, I2D* retDisp, int level, int nr, int nc);
//void cuda_integralImage2D2D(F2D* SAD, F2D* integralImg);
//void cuda_correlateSAD_2D(I2D* Ileft, I2D* Iright, I2D* Iright_moved, int win_sz, int disparity, F2D* SAD, F2D* integralImg, F2D* retSAD);
//kernels
__global__ void padarray4_kernel(I2D* inMat, int rows, int cols, int vborder, int hborder, int dir, I2D* paddedArray);
