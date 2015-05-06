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

F2D** diffss(F2D** ss, int num, int intervals, int gpu_transfer, int use_gpu)
{
    F2D** dss;
    int o, sizeM, sizeN, s, i, j;
    F2D *current, *in1, *in2;

    dss = (F2D**) malloc(num*intervals*sizeof(F2D*));

    for(o=0; o<num; o++)
    {
        for(s=0; s<(intervals-1); s++)
        {
            sizeM = ss[o*intervals+s]->height;
            sizeN = ss[o*intervals+s]->width;

            dss[o*intervals+s] = fMallocHandle(sizeM, sizeN);

            current = dss[o*intervals+s];
            in1 = ss[o*intervals+s+1];
            in2 = ss[o*intervals+s];

            F2D * d_in1, * d_in2, * d_out;

            if (gpu_transfer)
            {
                d_in1 = fMallocAndCopy(in1);
                d_in2 = fMallocAndCopy(in2);
                d_out = fMallocAndCopy(in2);
            }
            if (use_gpu)
            {
                dim3 dim_grid((sizeN - 1) / 32 + 1, (sizeM - 1) / 32 + 1, 1);
                dim3 dim_block(32, 32, 1);
                diffs_kernel<<<dim_grid, dim_block>>>(d_in1, d_in2, d_out);
                cudaMemcpy( current, d_out, sizeof(F2D)+sizeof(float)*current->height*current->width, cudaMemcpyDeviceToHost);
            }

            else
            {
                if (gpu_transfer)
                    cudaMemcpy( current, d_out, sizeof(F2D)+sizeof(float)*current->height*current->width, cudaMemcpyDeviceToHost);

                for(i=0; i<sizeM; i++)
                {
                    for(j=0; j<sizeN; j++)
                    {
                        subsref(current,i,j) = subsref(in1,i,j) - subsref(in2,i,j);
                    }
                }
            }

            if (gpu_transfer)
            {
                cudaFree(d_in1);
                cudaFree(d_in2);
                cudaFree(d_out);
            }
        }
    }

    return dss;

}




