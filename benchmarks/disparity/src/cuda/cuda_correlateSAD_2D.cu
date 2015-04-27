#include <stdio.h>
#include <stdlib.h>
#include "disparity.h"
#include "cuda_disparity.cu"

void cuda_correlateSAD_2D(I2D* Ileft, I2D* Iright, I2D* Iright_moved, int win_sz, int disparity, int rows, int cols, F2D* SAD, F2D* integralImg, F2D* retSAD) 
{
    
    int vborder = 0;
    int hborder = disparity;

    // setup execution parameters
    dim3  threads(32, 8, 1);
    int gridx = rows/threads.x + ((rows % threads.x == 0) ? 0:1);
    int gridy = cols/threads.x + ((cols % threads.y == 0) ? 0:1);
    dim3  griddim( gridx, gridy, 1);
    //printf("executing padarray %dx%d TB %dx%d grid, %dx%d Iright, %dx%d Iright_moved\n", threads.x, threads.y, gridx, gridy, rows, cols, rows+vborder, rows+hborder);
    padarray4_kernel<<<griddim, threads>>>(Iright, rows, cols, vborder, hborder, -1, Iright_moved);
    //printf("done padarray\n");

    cuda_computeSAD(Ileft, Iright_moved, SAD);
    cuda_integralImage2D2D(SAD, integralImg);
    cuda_finalSAD(integralImg, win_sz, retSAD);

    return;
}
