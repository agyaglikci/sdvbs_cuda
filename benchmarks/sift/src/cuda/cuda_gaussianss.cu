/********************************
Author: Sravanthi Kota Venkata
********************************/

#include "sift.h"
#include "cuda_sift.h"

F2D* resizeArray1(F2D* array, int omin)
{
    F2D* prev = NULL;
    F2D* current = array;
    int o;
    if(omin<0)
    {
        for(o=1; o>=-omin; o--)
        {
            prev = current;
            current = doubleSize(current);
            fFreeHandle(prev);
        }
    }
    if(omin>0)
    {
        for(o=1; o<= omin; o++)
        {
            prev = current;
            current = halveSize(current);
            fFreeHandle(prev);
        }
    }
    return current;
}

/**
    Returns the differnce of gaussian scale space of image I. Image I is assumed to be
    pre-smoothed at level SIGMAN. O,S,OMIN,SMIN,SMAX,SIGMA0 are the
    parameters of the scale space.
**/

F2D** cuda_gaussianss(F2D* array, float sigman, int O, int S, int omin, int smin, int smax, float sigma0)
{
    float k, dsigma0, dsigma;
    int s, i, j, o, so, M, N, sbest;
    int intervals = smax-smin+1;
    float temp, target_sigma, prev_sigma;
    F2D *TMP, **gss, **d_gss;
    F2D* I = array;

    k = pow(2, (1.0/S));
    dsigma0 = sigma0 * sqrt(1-(1.0/pow(k,2)));

    I = resizeArray1(I, omin);
    M = I->height;
    N = I->width;
    so = -smin+1;
    gss = (F2D**) malloc(O*intervals*sizeof(F2D*));
    d_gss = (F2D**) malloc(O*intervals*sizeof(F2D*));

    unsigned long long int transferTime = 0;
    struct timespec start, end;


    if(gss == NULL)
    {
        printf("Could not allocate memory\n");
        return NULL;
    }

    temp = sqrt(pow((sigma0*pow(k,smin)),2) - pow((sigman/pow(2,omin)),2));

    float * filter;
    gss[0] = fMallocHandle(I->height, I->width);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    cudaMalloc((void**) &filter, 1000 * sizeof(float));
    F2D * d_I = fMallocAndCopy(I);
    d_gss[0] = fMallocCudaArray(gss[0]);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    transferTime += 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;

    cuda_imsmooth(d_I, temp, d_gss[0], filter, gss[0]->width, gss[0]->height);
    cudaFree(d_I);
    fFreeHandle(I);


    for(s=smin; s<smax; s++)
    {
        dsigma = pow(k,s+1) * dsigma0;
        gss[s+so] = fMallocHandle(gss[s+so-1]->height, gss[s+so-1]->width);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
            d_gss[s+so] = fMallocCudaArray(gss[s+so]);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
        transferTime += 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
        cuda_imsmooth( d_gss[(s+so-1)] , dsigma, d_gss[(s+so)], filter, gss[s+so]->width, gss[s+so]->height);
    }

    for(o=1; o<O; o++)
    {
        sbest = MIN(smin+S-1, smax-1);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
            fCopyFromGPU(gss[(o-1)*intervals+sbest+so], d_gss[(o-1)*intervals+sbest+so]);
            TMP = halveSize( gss[(o-1)*intervals+sbest+so]);
            gss[o*intervals] = fMallocHandle(TMP->height, TMP->width);
            d_gss[o*intervals] = fMallocCudaArray(gss[o*intervals]);

            F2D * d_TMP = fMallocAndCopy(TMP);
            cudaMemcpy(d_gss[o*intervals], d_TMP, TMP->height*TMP->width*sizeof(float) + sizeof(F2D), cudaMemcpyDeviceToDevice);
            cudaFree(d_TMP);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
        transferTime += 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;

        // int halfWidth = (gss[(o-1)*intervals+sbest+so]->width + 1) / 2;
        // int halfHeight = (gss[(o-1)*intervals+sbest+so]->height + 1) / 2;
        // gss[o*intervals] = fMallocHandle(halfHeight, halfWidth);
        // d_gss[o*intervals] = fMallocCudaArray(gss[o*intervals]);

        // dim3 dim_grid((halfWidth - 1) / 32 + 1, (halfHeight - 1) / 32 + 1, 1);
        // dim3 dim_block(32, 32, 1);
        // dim3 dim_grid(1,1,1);
        // dim3 dim_block(1, 1, 1);
        // halfSize_kernel<<<dim_grid, dim_block>>>(d_gss[(o-1)*intervals+sbest+so], d_gss[o*intervals], halfWidth, halfHeight);


        fFreeHandle(TMP);
        for(s=smin; s<smax; s++)
        {
            // The other levels are determined as above for the first octave.
            dsigma = pow(k,s+1) * dsigma0;
            gss[o*intervals+s+so] = fMallocHandle(gss[o*intervals+s-1+so]->height, gss[o*intervals+s-1+so]->width);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

            d_gss[o*intervals+s+so] = fMallocCudaArray(gss[o*intervals+s+so]);

            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
            transferTime += 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
            cuda_imsmooth( d_gss[o*intervals+s-1+so] , dsigma, d_gss[o*intervals+s+so], filter,
                           gss[o*intervals+s+so]->width, gss[o*intervals+s+so]->height);
        }

    }

    cudaFree(filter);

    for(int o=0; o<O; o++)
    {
        for(int s=0; s<(intervals-1); s++)
        {
            int sizeM = gss[o*intervals+s]->height;
            int sizeN = gss[o*intervals+s]->width;

            dim3 dim_grid((sizeN - 1) / 32 + 1, (sizeM - 1) / 32 + 1, 1);
            dim3 dim_block(32, 32, 1);
            diffs_kernel<<<dim_grid, dim_block>>>(d_gss[o*intervals+s+1], d_gss[o*intervals+s], d_gss[o*intervals+s]);

            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
            fCopyAndFree(gss[o*intervals+s], d_gss[o*intervals+s]);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
            transferTime += 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
        }
    }

    free(d_gss);
    printf("Transfer time: %llu\n", transferTime);
    return gss;
}
