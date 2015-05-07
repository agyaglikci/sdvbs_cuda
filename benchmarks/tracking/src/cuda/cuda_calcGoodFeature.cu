/********************************
Author: Sravanthi Kota Venkata
********************************/

#include "tracking.h"

/** Computes lambda matrix, strength at each pixel
    
    det = determinant( [ IverticalEdgeSq IhorzVertEdge; IhorzVertEdge IhorizontalEdgeSq] ) ;
    tr = IverticalEdgeSq + IhorizontalEdgeSq;
    lamdba = det/tr;

    Lambda is the measure of the strength of pixel
    neighborhood. By strength we mean the amount of 
    edge information it has, which translates to
    sharp features in the image.

    Input:  Edge images - vertical and horizontal
            Window size (neighborhood size)
    Output: Lambda, strength of pixel neighborhood

    Given the edge images, we compute strength based
    on how strong the edges are within each neighborhood.

**/

F2D* cuda_calcGoodFeature(F2D* verticalEdgeImage, F2D* horizontalEdgeImage, int cols, int rows, int winSize, F2D** compare_array, bool use_gpu_lambda)
{
    unsigned int* timer;
    int phasei = 1;
    timer = cudaStartPhase();

    F2D* d_verticalEdgeSq = fMallocCudaArray(rows, cols);
    F2D* d_horzVertEdge = fMallocCudaArray(rows, cols);
    F2D* d_horizontalEdgeSq = fMallocCudaArray(rows, cols);
    F2D* d_cummulative_verticalEdgeSq = fMallocCudaArray(rows, cols);
    F2D* d_cummulative_horizontalEdgeSq = fMallocCudaArray(rows, cols);
    F2D* d_cummulative_horzVertEdge = fMallocCudaArray(rows, cols);

    F2D* areaSumTemp = fMallocCudaArray(rows, cols);
    F2D* lambda = fMallocHandle(rows, cols);

    cudaEndPhase(timer, phasei++, false);
    //printf("calcGoodFeature alloc gpu\n");
    timer = cudaStartPhase();

    dim3 blockdim(64,4,1);
    dim3 griddim(1,1,1);
    griddim.x = cols / blockdim.x + (cols % blockdim.x==0 ? 0 : 1);
    griddim.y = rows / blockdim.y + (rows % blockdim.y==0 ? 0 : 1);
    calcSquares_kernel<<<griddim, blockdim>>>(verticalEdgeImage, horizontalEdgeImage, cols, rows, d_verticalEdgeSq, d_horizontalEdgeSq, d_horzVertEdge);

    
    dim3 blockdim_row(64, 1, 1);
    dim3 griddim_row(1,1,1);
    griddim_row.x = cols / blockdim_row.x + (cols % blockdim_row.x==0 ? 0 : 1);
    griddim_row.y = 1;//rows / blockdim_row.y + (rows % blockdim_row.y==0 ? 0 : 1);
    dim3 blockdim_col(64, 1, 1);
    dim3 griddim_col(1, 1, 1);
    griddim_col.x = cols / blockdim_col.x + (cols % blockdim_col.x==0 ? 0 : 1);
    griddim_col.y = 1;//rows / blockdim_col.y + (rows % blockdim_col.y==0 ? 0 : 1);
    calcAreaSumRows_kernel<<<griddim_row, blockdim_row>>>(d_verticalEdgeSq, areaSumTemp, winSize);
    //printf("print areaSumTemp\n");
    //printSomeCuda(areaSumTemp, rows, cols);
    calcAreaSumCols_kernel<<<griddim_col, blockdim_col>>>(areaSumTemp, d_cummulative_verticalEdgeSq, winSize);
    calcAreaSumRows_kernel<<<griddim_row, blockdim_row>>>(d_horizontalEdgeSq, areaSumTemp, winSize);
    calcAreaSumCols_kernel<<<griddim_col, blockdim_col>>>(areaSumTemp, d_cummulative_horizontalEdgeSq, winSize);
    calcAreaSumRows_kernel<<<griddim_row, blockdim_row>>>(d_horzVertEdge, areaSumTemp, winSize);
    calcAreaSumCols_kernel<<<griddim_col, blockdim_col>>>(areaSumTemp, d_cummulative_horzVertEdge, winSize);
    GPUERRCHK;
    //printf("done with areasum\n");

    F2D* d_lambda;
    //F2D* d_tr = fMallocCudaArray(rows,cols);
    //F2D* d_det = fMallocCudaArray(rows,cols);

    if(use_gpu_lambda) 
    {
      F2D* d_lambda = fMallocCudaArray(rows,cols);
      blockdim.x = 32; blockdim.y = 4;
      griddim.x = cols / blockdim.x + (cols % blockdim.x==0 ? 0 : 1);
      griddim.y = rows / blockdim.y + (rows % blockdim.y==0 ? 0 : 1);
      calcLambda_kernel<<<griddim,blockdim>>>(d_cummulative_verticalEdgeSq, d_cummulative_horizontalEdgeSq, d_cummulative_horzVertEdge, cols, rows, d_lambda);
      GPUERRCHK;
      fCopyFromGPU(lambda, d_lambda);
      cudaFree(d_lambda);
      //printf("done with lamba gpu\n");
    }
    else 
    {
      F2D* cummulative_verticalEdgeSq = fMallocHandle(rows, cols);
      F2D* cummulative_horizontalEdgeSq = fMallocHandle(rows, cols);
      F2D* cummulative_horzVertEdge = fMallocHandle(rows, cols);
      fCopyFromGPU(cummulative_verticalEdgeSq, d_cummulative_verticalEdgeSq);
      fCopyFromGPU(cummulative_horizontalEdgeSq, d_cummulative_horizontalEdgeSq);
      fCopyFromGPU(cummulative_horzVertEdge, d_cummulative_horzVertEdge);

      F2D* tr = fMallocHandle(rows, cols);
      F2D* det = fMallocHandle(rows, cols);
      //printf("ready for lambdacalc\n");
      
      for(int i=0; i<rows; i++)
      {
          for(int j=0; j<cols; j++)
          {
              subsref(tr,i,j) = subsref(cummulative_verticalEdgeSq,i,j) + subsref(cummulative_horizontalEdgeSq,i,j);
              subsref(det,i,j) = subsref(cummulative_verticalEdgeSq,i,j) * subsref(cummulative_horizontalEdgeSq,i,j) - subsref(cummulative_horzVertEdge,i,j) * subsref(cummulative_horzVertEdge,i,j);
              subsref(lambda,i,j) = ( subsref(det,i,j) / (subsref(tr,i,j)+0.00001) ) ;
          }
      }
      fFreeHandle(tr);
      fFreeHandle(det);
      //printf("done with lambda cpu\n");
      
    }
    
    /*
    for( i=0; i<rows; i++)
    {
        for( j=0; j<cols; j++)
        {
            subsref(tr,i,j) = subsref(cummulative_verticalEdgeSq,i,j) + subsref(cummulative_horizontalEdgeSq,i,j);
            subsref(det,i,j) = subsref(cummulative_verticalEdgeSq,i,j) * subsref(cummulative_horizontalEdgeSq,i,j) - subsref(cummulative_horzVertEdge,i,j) * subsref(cummulative_horzVertEdge,i,j);
            subsref(lambda,i,j) = ( subsref(det,i,j) / (subsref(tr,i,j)+0.00001) ) ;
        }
    }*/
   
    cudaEndPhase(timer, phasei++, true);
    //printf("calcGoodFeature compute gpu\n");
    timer = cudaStartPhase();

    cudaFree(d_verticalEdgeSq);
    cudaFree(d_horzVertEdge);
    cudaFree(d_horizontalEdgeSq);
    cudaFree(d_cummulative_verticalEdgeSq);
    cudaFree(d_cummulative_horzVertEdge);
    cudaFree(d_cummulative_horizontalEdgeSq);
    
    //*compare_array = d_cummulative_verticalEdgeSq;
    //printf("print areaSumVertical\n");
    //printSomeCuda(d_cummulative_verticalEdgeSq, rows, cols);
     
    cudaEndPhase(timer, phasei++, false);
    //printf("calcGoodFeature free gpu\n");
    return lambda;
}
