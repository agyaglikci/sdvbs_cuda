/********************************
Author: Sravanthi Kota Venkata
********************************/

#include "sift.h"
#include <math.h>
//#include <assert.h>
const double win_factor = 1.5 ;
const int nbins = 36 ;
const float threshold = 0.01;

/**
    This function is similar to imageBlur in common/c folder.
    Here, we can specify the sigma value for the gaussian filter
    function.
**/

void imsmooth(F2D* array, float dsigma, F2D* out, int gpu_transfer)
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
    float temp[2*W+1];
    F2D* buffer;
    float acc = 0.0;

    buffer = fSetArray(M,N,0);

    if (gpu_transfer)
    {
          F2D * out2 = fMallocHandle(out->height, out->width);
          int filterSize = (2*W+1) * (2*W+1);
          float * gausFilter = (float *) malloc(sizeof(float)*filterSize);
          F2D* d_array = fMallocAndCopy(array);
          F2D* d_out = fMallocAndCopy(array);
          float * d_filt;
          cudaMalloc((void**) &d_filt, filterSize * sizeof(float));
          cudaMemcpy(d_filt, gausFilter, filterSize * sizeof(float), cudaMemcpyHostToDevice);

          fCopyFromGPU(out2, d_out);

          free(gausFilter);
          free(out2);
          cudaFree(d_array);
          cudaFree(d_out);
          cudaFree(d_filt);
    }


    for(j = 0 ; j < (2*W+1) ; ++j)
    {
      temp[j] = (float)(expf(-0.5 * (j - W)*(j - W)/(s*s))) ;
      acc += temp[j];
    }

    for(j = 0 ; j < (2*W+1) ; ++j)
    {
      temp[j] /= acc ;
    }

    /*
    ** Convolve along the columns
    **/

    for(j = 0 ; j < M ; ++j)
    {
      for(i = 0 ; i < N ; ++i)
      {
        int startCol = MAX(i-W,0);
        int endCol = MIN(i+W, N-1);
        int filterStart = MAX(0, W-i);

		//assert(j < array->height);
		//assert(j < buffer->height);
		//assert(i < buffer->width);
        for(k=startCol; k<=endCol; k++) {
			//assert(k < array->width);
			//assert(filterStart < 2*W+1);
            subsref(buffer,j,i) += subsref(array, j, k) * temp[filterStart++];
		}
      }
    }

    /*
    ** Convolve along the rows
    **/
    for(j = 0 ; j < M ; ++j)
    {
      for(i = 0 ; i < N ; ++i)
      {
        int startRow = MAX(j-W,0);
        int endRow = MIN(j+W, M-1);
        int filterStart = MAX(0, W-j);
        for(k=startRow; k<=endRow; k++)
            subsref(out,j,i) += subsref(buffer,k,i) * temp[filterStart++];
      }
    }

    fFreeHandle(buffer);


    /*int filterSize = (2*W+1) * (2*W+1);
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

    F2D * out2 = fMallocHandle(out->height, out->width);

    for(row=0; row<M; row++)
    {
      for (col=0; col<N; col++)
      {
        sum = 0.0f;
        for(int gRow=-radius; gRow <=radius; gRow++)
        {
          int offY = row + gRow;
          for (int gCol = -radius; gCol<=radius; gCol++)
          {
            int offX = col + gCol;
            if (offY<0||offX<0||offY>=M||offX>=N)
              continue;
            sum += array->data[offY*N + offX] *  gausFilter[(gRow+radius)*(2*W + 1)+(gCol+radius)];
          }
        }
        out2->data[row*N + col] = sum;
      }
    }

    for(row=0; row<M; row++)
    {
      for (col=0; col<N; col++)
      {
        out->data[row * N + col] = out2->data[row * N + col];
        if (out->data[row * N + col] != out2->data[row * N + col])
          printf("%f %f\n", out->data[row * N + col], out2->data[row * N + col]);
      }
    }

    free(gausFilter);
    fFreeHandle(out2);*/


  }
  else
  {
      for(i=0;i<M*N;i++)
          asubsref(out, i) = asubsref(array, i);
  }


  return;
}
