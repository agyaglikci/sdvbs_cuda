/********************************
Author: Sravanthi Kota Venkata
********************************/

#include <stdio.h>
#include <stdlib.h>
#include "sift.h"

bool use_gpu;
bool gpu_transfer;

int main(int argc, char* argv[])
{
    I2D* im;
    F2D *image;
    int rows, cols;
    F2D* frames;
    unsigned int* startTime, *endTime, *elapsed;

    char imSrc[100];

    if(argc < 2)
    {
        printf("We need input image path\n");
        return -1;
    }

    sprintf(imSrc, "%s/1.bmp", argv[1]);

    /*** ADD THIS TO EXISTING BENCHMARKS *****/
    use_gpu = atoi(argv[3]);
    gpu_transfer = atoi(argv[4]);
    /*****************************************/

    im = readImage(imSrc);
    image = fiDeepCopy(im);
    iFreeHandle(im);
    rows = image->height;
    cols = image->width;

    startTime = photonStartTiming();

    /** Extract sift features for the normalized image **/
    frames = sift(image, use_gpu, gpu_transfer);
    endTime = photonEndTiming();

    printf("Input size\t\t- (%dx%d)\n", rows, cols);

#ifdef CHECK
    {
        int ret=0;
        float tol = 0.2;
#ifdef GENERATE_OUTPUT
        fWriteMatrix(frames, argv[1]);
#endif
        ret = fSelfCheck(frames, argv[1], tol);
        if (ret == -1)
            printf("Error in SIFT\n");
    }
#endif

    elapsed = photonReportTiming(startTime, endTime);
    photonPrintTiming(elapsed);

    free(startTime);
    free(endTime);
    free(elapsed);

    fFreeHandle(frames);

    return 0;
}
