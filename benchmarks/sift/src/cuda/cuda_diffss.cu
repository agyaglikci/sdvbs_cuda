/********************************
Author: Sravanthi Kota Venkata
********************************/

#include "sift.h"
#include "cuda_sift.h"

/**
    DIFFSS  Difference of scale space
    Returns a scale space DSS obtained by subtracting
    consecutive levels of the scale space SS.

    In SIFT, this function is used to compute the difference of
    Gaussian scale space from the Gaussian scale space of an image.
**/

void cuda_diffss(F2D** ss, F2D** dss, int num, int intervals, int width, int height)
{
    int sizeM, sizeN;
    F2D *current;

    F2D * diff = fMallocCudaArray(width, height);

    for(int o=0; o<num; o++)
    {
        for(int s=0; s<(intervals-1); s++)
        {
            sizeM = dss[o*intervals+s]->height;
            sizeN = dss[o*intervals+s]->width;
            current = dss[o*intervals+s];
            printf("Index: %d %dx%d\n", o*intervals+s, sizeM, sizeN);

            dim3 dim_grid((sizeN - 1) / 32 + 1, (sizeM - 1) / 32 + 1, 1);
            dim3 dim_block(32, 32, 1);
            diffs_kernel<<<dim_grid, dim_block>>>(ss[o*intervals+s+1], ss[o*intervals+s], ss[o*intervals+s]);
            cudaMemcpy( dss[o*intervals+s], ss[o*intervals+s], sizeof(F2D)+sizeof(float)*current->width*current->height, cudaMemcpyDeviceToHost);
        }
    }
}




