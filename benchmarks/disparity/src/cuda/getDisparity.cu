/********************************
Author: Sravanthi Kota Venkata
********************************/

#include <stdio.h>
#include <stdlib.h>
#include "disparity.h"
//#include "cuda_disparity.cu"

/*
#define GPUERRCHK { gpuAssert((cudaGetLastError()), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
*/

I2D* getDisparity(I2D* Ileft, I2D* Iright, int win_sz, int max_shift, int use_gpu, int gpu_transfer)
{
    I2D* retDisp;
    int nr, nc, k;
    I2D *halfWin;
    int half_win_sz, rows, cols;
    F2D *retSAD, *minSAD, *SAD, *integralImg;
    I2D* IrightPadded, *IleftPadded, *Iright_moved;
    // device-side copies
    I2D* d_retDisp;
    F2D *d_retSAD, *d_minSAD, *d_SAD, *d_integralImg;
    I2D* d_IrightPadded, *d_IleftPadded, *d_Iright_moved;
    
    nr = Ileft->height;
    nc = Ileft->width;
    half_win_sz=win_sz/2;
    
    
    minSAD = fSetArray(nr, nc, 255.0*255.0);
    retDisp = iSetArray(nr, nc,max_shift);
    halfWin = iSetArray(1,2,half_win_sz);

        if(win_sz > 1)
        {
            IleftPadded = padarray2(Ileft, halfWin);
            IrightPadded = padarray2(Iright, halfWin);
        }
        else
        {
            IleftPadded = Ileft;
            IrightPadded = Iright;
        }
    
    rows = IleftPadded->height;
    cols = IleftPadded->width;
    SAD = fSetArray(rows, cols,255);
    integralImg = fSetArray(rows, cols,0);
    retSAD = fMallocHandle(rows-win_sz, cols-win_sz);
    Iright_moved = iSetArray(rows, cols, 0);

    if(gpu_transfer) 
    {
      unsigned int* start_transfer = cudaStartTransfer();
      printf("doing gpu transfer\n");
      d_IleftPadded = iMallocAndCopy(IleftPadded);
      d_IrightPadded = iMallocAndCopy(IrightPadded);
      d_Iright_moved = iMallocAndCopy(Iright_moved);
      d_SAD = fMallocAndCopy(SAD);
      d_integralImg = fMallocAndCopy(integralImg);
      d_retSAD = fMallocAndCopy(retSAD);
      d_minSAD = fMallocAndCopy(minSAD);
      d_retDisp = iMallocAndCopy(retDisp);
      GPUERRCHK;
      cudaEndTransfer(start_transfer);
    }
    
    for( k=0; k<max_shift; k++)
    {    
        if(use_gpu) 
        {
          cuda_correlateSAD_2D(d_IleftPadded, d_IrightPadded, d_Iright_moved, win_sz, k, nr, nc, SAD, integralImg, retSAD);
          cuda_findDisparity(d_retSAD, d_minSAD, d_retDisp, k, nr, nc);
          GPUERRCHK;
        }
        else
        {
          correlateSAD_2D(IleftPadded, IrightPadded, Iright_moved, win_sz, k, SAD, integralImg, retSAD);
          findDisparity(retSAD, minSAD, retDisp, k, nr, nc);
        }
    }
    //printf("done with kernel\n");
    
    fFreeHandle(retSAD);
    fFreeHandle(minSAD);
    fFreeHandle(SAD);
    fFreeHandle(integralImg);
    iFreeHandle(halfWin);
    iFreeHandle(IrightPadded);
    iFreeHandle(IleftPadded);
    iFreeHandle(Iright_moved);
     
    if(gpu_transfer) 
    {
      unsigned int* start_transfer = cudaStartTransfer();
      cudaFree(d_retSAD);
      cudaFree(d_minSAD);
      cudaFree(d_SAD);
      cudaFree(d_integralImg);
      cudaFree(d_IrightPadded);
      cudaFree(d_IleftPadded);
      cudaFree(d_Iright_moved);
      if(use_gpu) {
        iCopyFromGPU(retDisp, d_retDisp);
      }
      GPUERRCHK;
      cudaEndTransfer(start_transfer);
    }
    return retDisp;
}

