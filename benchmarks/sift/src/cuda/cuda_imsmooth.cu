/********************************
Author: Sravanthi Kota Venkata
********************************/

#include "sift.h"
#include <math.h>
#include "cuda_sift.h"

//#include <assert.h>
const double win_factor = 1.5 ;
const int nbins = 36 ;
const float threshold = 0.01;

/**
    This function is similar to imageBlur in common/c folder.
    Here, we can specify the sigma value for the gaussian filter
    function.
**/

void cuda_imsmooth(F2D* array, float dsigma, F2D* out)
{
  int M,N ;
  int i,j,k;
  float s ;

  /* ------------------------------------------------------------------
  **                                                Check the arguments
  ** --------------------------------------------------------------- */

    M = array->height;
    N = array->width;
    s = dsigma;

  /* ------------------------------------------------------------------
  **                                                         Do the job
  ** --------------------------------------------------------------- */
  if(s > threshold)
  {
    int W = (int) ceil(4*s) ;
    int filterSize = (2*W+1) * (2*W+1);
    float * gausFilter = (float *) malloc(sizeof(float)*filterSize);
    float sum = 0.0f;
    int radius = W;
    double tempVal = sqrt(2*M_PI*s*s);
    int row, col, idx =0;
    for(row = -radius; row <= radius; row++)
    {
      for(col = -radius; col <= radius; col++)
      {
        gausFilter[idx] = (float) (exp(-(row*row+col*col)/(2*s*s))/tempVal);
        sum+=gausFilter[idx++];
      }
    }
    int i;
    for (i=0; i<idx;i++)
      gausFilter[i] /= sum;

    // for(row=0; row<M; row++)
    // {
    //   for (col=0; col<N; col++)
    //   {
    //     sum = 0.0f;
    //     for(int gRow=-radius; gRow <=radius; gRow++)
    //     {
    //       int offY = row + gRow;
    //       for (int gCol = -radius; gCol<=radius; gCol++)
    //       {
    //         int offX = col + gCol;
    //         if (offY<0||offX<0||offY>=M||offX>=N)
    //           continue;
    //         sum += array->data[offY*N + offX] *  gausFilter[(gRow+radius)*(2*W + 1)+(gCol+radius)];
    //       }
    //     }
    //     out->data[row*N + col] = sum;
    //   }
    // }

    F2D* d_array = fMallocAndCopy(array);
    F2D* d_out = fMallocAndCopy(array);
    float * d_filt;
    cudaMalloc((void**) &d_filt, filterSize * sizeof(float));
    cudaMemcpy(d_filt, gausFilter, filterSize * sizeof(float), cudaMemcpyHostToDevice);

    int outTileSize = 6;
    int inTileSize = outTileSize + 2 * W;

    dim3 dim_grid((N - 1) / outTileSize + 1, (M - 1) / outTileSize + 1, 1);
    dim3 dim_block(inTileSize, inTileSize, 1);

    imsmooth_kernel<<<dim_grid, dim_block, inTileSize * inTileSize * sizeof(float)>>>(d_array, d_out, 2*W + 1, inTileSize, outTileSize, d_filt);


    fCopyFromGPU(out, d_out);

    // for(row=0; row<M; row++)
    // {
    //   for (col=0; col<N; col++)
    //   {
    //     printf("%f %f\n", out->data[row*N + col], out2->data[row*N + col]);
    //   }
    // }

    // printf("%s: %d\n", __FILE__, __LINE__);

    // // printf("%s: %d\n", __FILE__, __LINE__);
    // // for(row=0; row<M; row++)
    // // {
    // //   for (col=0; col<N; col++)
    // //   {
    // //     printf("%f\n", out->data[row*N + col]);
    // //   }
    // // }
    free(gausFilter);

    cudaFree(d_array);
    cudaFree(d_out);
    cudaFree(d_filt);

  }
  else
  {
      for(i=0;i<M*N;i++)
          asubsref(out, i) = asubsref(array, i);
  }


  return;
}
