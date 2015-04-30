#include <stdio.h>
#include <stdlib.h>
#include "disparity.h"
#include "cuda_disparity.h"

void cuda_correlateSAD_2D(I2D* Ileft, I2D* Iright, I2D* Iright_moved, int win_sz, int disparity, int rows, int cols, F2D* SAD, F2D* integralImg, F2D* retSAD)
{

    int vborder = 0;
    int hborder = disparity;

    int padrows = rows;//+win_sz;
    int padcols = cols;//+win_sz;

    // setup execution parameters
    dim3  threads(64, 4, 1);
    int gridx = padcols/threads.x + ((padcols % threads.x == 0) ? 0:1);
    int gridy = padrows/threads.y + ((padrows % threads.y == 0) ? 0:1);
    dim3  griddim( gridx, gridy, 1);
    //printf("executing padarray %dx%d TB %dx%d grid, %dx%d Iright, %dx%d Iright_moved\n", threads.x, threads.y, gridx, gridy, rows, cols, padrows, padcols);
    padarray4_kernel<<<griddim, threads>>>(Iright, vborder, hborder, -1, Iright_moved);
//#ifdef DEBUG
//    I2D* h_Iright_moved = iMallocHandle(padrows, padcols);
//    iCopyFromGPU(h_Iright_moved, Iright_moved);
//    printf("gpu padarray:\n");
//    printSome(h_Iright);
//#endif
//#ifdef DEBUG
//    I2D* h_Ileft = iMallocHandle(padrows, padcols);
//    iCopyFromGPU(h_Ileft, Ileft);
//    printf("gpu Ileft:\n");
//    printSome(h_Ileft);
//#endif

    computeSAD_kernel<<<griddim, threads>>>(Ileft, Iright_moved, SAD);
//#ifdef DEBUG
//    F2D* h_SAD = fMallocHandle(padrows, padcols);
//    fCopyFromGPU(h_SAD, SAD);
//    printf("gpu computeSAD:\n");
//    printSome(h_SAD);
//#endif

    threads.x = 64; threads.y=1;
    griddim.x = padcols/threads.x + ((padcols % threads.x == 0) ? 0:1);
    griddim.y = 1;
    //printf("launching hintegralImg  %dx%d TB %dx%d grid, %dx%d\n", threads.x, threads.y, griddim.x, griddim.y);
    hintegralImage2D2D_kernel<<<griddim, threads>>>(SAD, integralImg);
//#ifdef DEBUG
//    F2D* h_integralImg = fMallocHandle(padrows, padcols);
//    fCopyFromGPU(h_integralImg, integralImg);
//    printf("gpu integralImg1:\n");
//    printSome(h_integralImg);
//#endif
    griddim.x = padrows/threads.x + ((padrows % threads.x == 0) ? 0:1);
    vintegralImage2D2D_kernel<<<griddim, threads>>>(SAD, integralImg);
//#ifdef DEBUG
//    F2D* h_integralImg = fMallocHandle(padrows, padcols);
//    fCopyFromGPU(h_integralImg, integralImg);
//    printf("gpu integralImg2:\n");
//    printSome(h_integralImg);
//#endif

    threads.x = 32; threads.y=8;
    griddim.x = padcols/threads.x + ((padcols % threads.x == 0) ? 0:1);
    griddim.y = padrows/threads.y + ((padrows % threads.y == 0) ? 0:1);
    //printf("%dx%dx%d threads, %dx%dx%d grid, %dx%d retSAD\n", threads.x, threads.y, threads.z, griddim.x, griddim.y, griddim.z, padcols, padrows);
    GPUERRCHK;
    finalSAD_kernel<<<griddim, threads>>>(integralImg, win_sz, retSAD);
    GPUERRCHK;
    return;
}
