/********************************
Author: Sravanthi Kota Venkata
********************************/

#include "sift.h"
#include <math.h>
#include "cuda_sift.h"
#include <sys/time.h>

//#include <assert.h>
const double win_factor = 1.5 ;
const int nbins = 36 ;
const float threshold = 0.01;

/**
    This function is similar to imageBlur in common/c folder.
    Here, we can specify the sigma value for the gaussian filter
    function.
**/

void cuda_imsmooth(F2D* array, float dsigma, F2D* out, float * d_filt, int width, int height)
{
  int i, j, k;
  float s = dsigma;

  if(s > threshold)
  {
    int filterRadius = (int) ceil(4*s);
    int filterLength = 2 * filterRadius + 1;
    float * filter = (float *) malloc(filterLength*sizeof(float));
    float sum = 0.0;
    for(j = 0; j < filterLength; j++)
    {
      filter[j] = (float)(expf(-0.5 * (j - filterRadius)*(j - filterRadius)/(s*s))) ;
      sum += filter[j];
    }

    for(j = 0; j < filterLength; j++)
    {
      filter[j] /= sum ;
      //printf("%f ", filter[j]);
    }
    //printf("\n");

    int tileW = 16;
    int tileH = 16;
    dim3 blocks(tileW, tileH);
    //printf("blocks (%d, %d)\n", blocks.x, blocks.y);
    dim3 grids(width/tileW, height/tileH);
    //printf("grids  (%d, %d)\n", grids.x, grids.y);

    F2D * arrayB = fMallocCudaArray(width, height);
    cudaMemcpy(d_filt, filter, filterLength * sizeof(float), cudaMemcpyHostToDevice);
    printf("filterRadius: %d width: %d height: %d\n", filterRadius, width, height);
    imsmoothRow_kernel<<<grids, blocks, tileH * (tileW + filterRadius*2) * sizeof(float)>>>(arrayB, array, filterRadius, tileW, d_filt, width, height);
    imsmoothCol_kernel<<<grids, blocks, tileW * (tileH + filterRadius*2) * sizeof(float)>>>(out, arrayB, filterRadius, tileW, d_filt, width, height);
    free(filter);

    F2D * temp = (F2D *) malloc(sizeof(F2D) + width * height * sizeof(float));
    cudaMemcpy(temp, out, sizeof(F2D) + width * height * sizeof(float), cudaMemcpyDeviceToHost);
    //printf("%d %d\n", temp->width, temp->height);
    printf("Separ: %f %f\n", temp->data[0], temp->data[10]);

    cudaFree(arrayB);

    int W = (int) ceil(4*s) ;
    int filterSize = (2*W+1) * (2*W+1);
    float * gausFilter = (float *) malloc(sizeof(float)*filterSize);

    //float sum = 0.0f;
    sum = 0.0f;
    int radius = W;
    double tempVal = sqrt(2*M_PI*s*s);
    int row, col, idx =0;
    int kernelWidth = 2*W + 1;

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


    cudaMemcpy(d_filt, gausFilter, filterSize * sizeof(float), cudaMemcpyHostToDevice);

    int outTileSize = 6;
    int inTileSize = outTileSize + 2 * W;

    dim3 dim_grid((width - 1) / outTileSize + 1, (height - 1) / outTileSize + 1, 1);
    dim3 dim_block(inTileSize, inTileSize, 1);

    imsmooth_kernel<<<dim_grid, dim_block, inTileSize * inTileSize * sizeof(float)>>>(array, out, 2*W + 1, inTileSize, outTileSize, d_filt);
    free(gausFilter);

    cudaMemcpy(temp, out, sizeof(F2D) + width * height * sizeof(float), cudaMemcpyDeviceToHost);
    //printf("%d %d\n", temp->width, temp->height);
    printf("Joint: %f %f\n", temp->data[0], temp->data[10]);

    // free(temp);
  }
  else
    cudaMemcpy(out, array, sizeof(F2D) + array->width * array->height * sizeof(float), cudaMemcpyDeviceToDevice);

}
