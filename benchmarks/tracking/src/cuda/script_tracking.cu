/********************************
Author: Sravanthi Kota Venkata
********************************/
//#undef DEBUG
#include "tracking.h"

bool use_gpu;
bool gpu_transfer;

int main(int argc, char* argv[])
{
    int i, j, k, N_FEA, WINSZ, LK_ITER, rows, cols;
    int endR, endC;
    F2D *blurredImage, *previousFrameBlurred_level1, *previousFrameBlurred_level2, *blurred_level1, *blurred_level2;
    F2D *d_blurredImage, *d_previousFrameBlurred_level1, *d_previousFrameBlurred_level2, *d_blurred_level1, *d_blurred_level2, *d_blurred_temp;
    F2D *verticalEdgeImage, *horizontalEdgeImage, *verticalEdge_level1, *verticalEdge_level2, *horizontalEdge_level1, *horizontalEdge_level2, *interestPnt;
    F2D *d_verticalEdgeImage, *d_horizontalEdgeImage, *d_verticalEdge_level1, *d_verticalEdge_level2, *d_horizontalEdge_level1, *d_horizontalEdge_level2, *d_interestPnt;
    F2D *d_edgeTemp, *d_edgeTemp_level2;
    F2D *lambda, *lambdaTemp, *features;
    I2D *Ic, *status;
    float SUPPRESION_RADIUS;
    F2D *newpoints;

    int numFind, m, n;
    F2D *np_temp;

    unsigned int* start, *end, *elapsed, *elt;
    char im1[100];
    int counter=2;
    float accuracy = 0.03;
    int count;
    
    if(argc < 2) 
    {
        printf("We need input image path\n");
        return -1;
    }

    sprintf(im1, "%s/1.bmp", argv[1]);

    N_FEA = 1600;
    WINSZ = 4;
    SUPPRESION_RADIUS = 10.0;
    LK_ITER = 20;

    #ifdef test
        WINSZ = 2;
        N_FEA = 100;
        LK_ITER = 2;
        counter = 2;
        accuracy = 0.1;
    #endif
    #ifdef sim_fast
        WINSZ = 2;
        N_FEA = 100;
        LK_ITER = 2;
        counter = 4;
    #endif
    #ifdef sim
        WINSZ = 2;
        N_FEA = 200;
        LK_ITER = 2;
        counter = 4;
    #endif
    #ifdef sqcif
        WINSZ = 8;
        N_FEA = 500;
        LK_ITER = 15;
        counter = 2;
    #endif
    #ifdef qcif
        WINSZ = 12;
        N_FEA = 400;
        LK_ITER = 15;
        counter = 4;
    #endif
    #ifdef cif
        WINSZ = 20;
        N_FEA = 500;
        LK_ITER = 20;
        counter = 4;
    #endif
    #ifdef vga
        WINSZ = 32;
        N_FEA = 400;
        LK_ITER = 20;
        counter = 4;
    #endif
    #ifdef wuxga
        WINSZ = 64;
        N_FEA = 500;
        LK_ITER = 20;
        counter = 4;
    #endif
    #ifdef fullhd
        WINSZ = 48;
        N_FEA = 500;
        LK_ITER = 20;
        counter = 4;
    #endif

    /*** ADD THIS TO EXISTING BENCHMARKS *****/
    use_gpu = atoi(argv[3])==1;
    gpu_transfer = atoi(argv[4])==1;
    /*****************************************/
    bool use_cpu = !use_gpu;//true;
    bool use_gpu_lambda = use_gpu;//!use_gpu;//true;

    /** Read input image **/
    Ic = readImage(im1);
    rows = Ic->height;
    cols = Ic->width;
    
    printf("Input size\t\t- (%dx%d)\n", rows, cols);
    
    /** Start Timing **/
    start = photonStartTiming();

    int phasei=1;
    unsigned int* timer_start;
    unsigned int* timer;
    timer_start = cudaStartPhase();
    timer = cudaStartPhase();
    /** IMAGE PRE-PROCESSING **/

    /** Blur the image to remove noise - weighted avergae filter **/
    if(use_cpu) {
      blurredImage = imageBlur(Ic);
    }

    I2D* d_Ic;
    F2D* d_tempImage;
    dim3 blockdim(32, 8, 1);
    dim3 griddim(1, 1, 1);
    int resizeRows = floor((rows+1)/2);
    int resizeCols = floor((cols+1)/2);
    if(use_gpu) 
    {
      d_Ic = iMallocAndCopy(Ic);
      d_tempImage = fMallocCudaArray(rows, cols);
      d_blurredImage = fMallocCudaArray(rows, cols);

      d_blurred_temp = fMallocCudaArray(rows, resizeCols);
      d_blurred_level2 = fMallocCudaArray(resizeRows, resizeCols);

      d_verticalEdgeImage = fMallocCudaArray(rows, cols);
      d_edgeTemp = fMallocCudaArray(rows, cols);
      d_horizontalEdgeImage = fMallocCudaArray(rows, cols);

      cudaEndPhase(timer, phasei++, false);
      //printf("malloc/copy gpu\n");
      timer = cudaStartPhase();
      griddim.x = cols / blockdim.x + (cols % blockdim.x == 0 ? 0 : 1);
      griddim.y = rows / blockdim.y + (rows % blockdim.y == 0 ? 0 : 1);
      imageBlurHorizontal_kernel<<<griddim, blockdim>>>(d_Ic, d_tempImage);
      GPUERRCHK;
      imageBlurVertical_kernel<<<griddim, blockdim>>>(d_tempImage, d_blurredImage);
      GPUERRCHK;

#ifdef DEBUG
      F2D* blurredImage_debug = fMallocHandle(rows, cols);
      GPUERRCHK;
      printf("comparing blurredImage\n");
      fCopyFromGPU(blurredImage_debug, d_blurredImage);
      GPUERRCHK;
      compareArrays(blurredImage, blurredImage_debug);
#endif
    }

    /** Scale down the image to build Image Pyramid. We find features across all scales of the image **/
    blurred_level1 = blurredImage;                   /** Scale 0 **/
    if(use_cpu) 
    {
      blurred_level2 = imageResize(blurredImage);      /** Scale 1 **/
    }
    cudaEndPhase(timer, phasei++, true);
    //printf("blur done\n");
    timer = cudaStartPhase();

    d_blurred_level1 = d_blurredImage;
    if(use_gpu) {

      GPUERRCHK;
      griddim.x = resizeCols / blockdim.x + (resizeCols % blockdim.x == 0 ? 0 : 1);
      //printf("launching imageResizeHorizontal_kernel with grid %dx%dx%d and tb %dx%dx%d\n", griddim.x, griddim.y, griddim.z, blockdim.x, blockdim.y, blockdim.z);
      imageResizeHorizontal_kernel<<<griddim, blockdim>>>(d_blurred_level1, d_blurred_temp);
      griddim.y = resizeRows / blockdim.y + (resizeRows % blockdim.y == 0 ? 0 : 1);
      //printf("launching imageResizeVertical_kernel with grid %dx%dx%d and tb %dx%dx%d\n", griddim.x, griddim.y, griddim.z, blockdim.x, blockdim.y, blockdim.z);
      imageResizeVertical_kernel<<<griddim, blockdim>>>(d_blurred_temp, d_blurred_level2);
      GPUERRCHK;

#ifdef DEBUG
      //printf("comparing blurred_level2\n");
      F2D* blurred_level2_debug = fMallocHandle(resizeRows, resizeCols);
      GPUERRCHK;
      fCopyFromGPU(blurred_level2_debug, d_blurred_level2);
      GPUERRCHK;
      compareArrays(blurred_level2, blurred_level2_debug);
#endif
    }

    cudaEndPhase(timer, phasei++, true);
    //printf("resize done\n");
    timer = cudaStartPhase();

    /** Edge Images - From pre-processed images, build gradient images, both horizontal and vertical **/
    if(use_cpu) {
      verticalEdgeImage = calcSobel_dX(blurredImage);
      horizontalEdgeImage = calcSobel_dY(blurredImage);
    }

    if(use_gpu) 
    {
      griddim.y = rows / blockdim.y + (rows % blockdim.y == 0 ? 0 : 1);
      griddim.x = cols / blockdim.x + (cols % blockdim.x == 0 ? 0 : 1);
      //printf("launching calcSobelHor with grid %dx%dx%d and tb %dx%dx%d\n", griddim.x, griddim.y, griddim.z, blockdim.x, blockdim.y, blockdim.z);
      calcSobelHorizontal_dX_kernel<<<griddim, blockdim>>>(d_blurredImage, d_edgeTemp);
      //printf("launching calcSobelVert with grid %dx%dx%d and tb %dx%dx%d\n", griddim.x, griddim.y, griddim.z, blockdim.x, blockdim.y, blockdim.z);
      calcSobelVertical_dX_kernel<<<griddim, blockdim>>>(d_edgeTemp, d_verticalEdgeImage);
      GPUERRCHK;

#ifdef DEBUG
      printf("comparing verticalEdge\n");
      F2D* verticalEdge_debug = fMallocHandle(rows, cols);
      GPUERRCHK;
      fCopyFromGPU(verticalEdge_debug, d_edgeTemp);
      fCopyFromGPU(verticalEdge_debug, d_verticalEdgeImage);
      GPUERRCHK;
      compareArrays(verticalEdgeImage, verticalEdge_debug);
#endif


      //printf("launching calcSobelHor with grid %dx%dx%d and tb %dx%dx%d\n", griddim.x, griddim.y, griddim.z, blockdim.x, blockdim.y, blockdim.z);
      calcSobelVertical_dY_kernel<<<griddim, blockdim>>>(d_blurredImage, d_edgeTemp);
      calcSobelHorizontal_dY_kernel<<<griddim, blockdim>>>(d_edgeTemp, d_horizontalEdgeImage);
      //printf("launching calcSobelVert with grid %dx%dx%d and tb %dx%dx%d\n", griddim.x, griddim.y, griddim.z, blockdim.x, blockdim.y, blockdim.z);
      GPUERRCHK;
    
#ifdef DEBUG
      printf("comparing horizontalEdge\n");
      F2D* horizontalEdge_debug = fMallocHandle(rows, cols);
      GPUERRCHK;
      fCopyFromGPU(horizontalEdge_debug, d_horizontalEdgeImage);
      GPUERRCHK;
      compareArrays(horizontalEdgeImage, horizontalEdge_debug);
#endif
    }

    cudaEndPhase(timer, phasei++, true);
    //printf("edges done\n");

    /** Edge images are used for feature detection. So, using the verticalEdgeImage and horizontalEdgeImage images, we compute feature strength
        across all pixels. Lambda matrix is the feature strength matrix returned by calcGoodFeature **/
    
    F2D* compare_array, *d_compare_array;
    if(use_cpu) 
    {
      lambda = calcGoodFeature(verticalEdgeImage, horizontalEdgeImage, verticalEdgeImage->width, verticalEdgeImage->height, WINSZ, &compare_array);
    }
  
    F2D* d_lambda;
    if(use_gpu)
    {
      lambda = cuda_calcGoodFeature(d_verticalEdgeImage, d_horizontalEdgeImage, cols, rows, WINSZ, &d_compare_array, use_gpu_lambda);

#ifdef DEBUG
/*
      printf("comparing compare-array\n");
      F2D* compare_array_debug = fMallocHandle(rows, cols);
      GPUERRCHK;
      fCopyFromGPU(compare_array_debug, d_compare_array);
      GPUERRCHK;
      compareArrays(compare_array, compare_array_debug);
      printf("comparing lambda\n");
      F2D* lambda_debug = fMallocHandle(rows, cols);
      GPUERRCHK;
      fCopyFromGPU(lambda_debug, d_lambda);
      GPUERRCHK;
      compareArrays(lambda, lambda_debug);
*/
#endif
    }
    timer = cudaStartPhase();
    endR = lambda->height;
    endC = lambda->width;
    lambdaTemp = fReshape(lambda, endR*endC, 1);

    cudaEndPhase(timer, phasei++);
    //printf("got lambda\n");
    timer = cudaStartPhase();

    /** We sort the lambda matrix based on the strengths **/
    /** Fill features matrix with top N_FEA features **/
    fFreeHandle(lambdaTemp);
    lambdaTemp = fillFeatures(lambda, N_FEA, WINSZ);
    features = fTranspose(lambdaTemp);

    cudaEndPhase(timer, phasei++);
    //printf("sorted lambda\n");
    timer = cudaStartPhase();
    
    /** Suppress features that have approximately similar strength and belong to close neighborhood **/
    interestPnt = getANMS(features, SUPPRESION_RADIUS);

    cudaEndPhase(timer, phasei++);
    //printf("got ANMS\n");
    timer = cudaStartPhase();
    
    /** Refill interestPnt in features matrix **/
    fFreeHandle(features);
    features = fSetArray(2, interestPnt->height, 0);
    for(i=0; i<2; i++) {
        for(j=0; j<interestPnt->height; j++) {
            subsref(features,i,j) = subsref(interestPnt,j,i); 
		}
    } 
     
    cudaEndPhase(timer, phasei++);
    //printf("got features\n");
    timer = cudaStartPhase();

    end = photonEndTiming();
    elapsed = photonReportTiming(start, end);

    if(use_cpu) 
    {
      fFreeHandle(verticalEdgeImage);
      fFreeHandle(horizontalEdgeImage);
    }
  
    fFreeHandle(interestPnt);
    fFreeHandle(lambda);
    fFreeHandle(lambdaTemp);
    iFreeHandle(Ic);
    free(start);
    free(end);
    // no compute there
    cudaEndPhase(timer, phasei++, false);
    cudaEndPhase(timer_start, 0, false, false);
    //printf("start loop\n");
    timer = cudaStartPhase();

/** Until now, we processed base frame. The following for loop processes other frames **/
for(count=1; count<=counter; count++)
{
    /** Read image **/
    sprintf(im1, "%s/%d.bmp", argv[1], count);
    Ic = readImage(im1);
    rows = Ic->height;
    cols = Ic->width;
    
    /* Start timing */
    start = photonStartTiming();

    /** Blur image to remove noise **/
    if(use_cpu)
    {
      previousFrameBlurred_level1 = fDeepCopy(blurred_level1);
      previousFrameBlurred_level2 = fDeepCopy(blurred_level2);
      
      fFreeHandle(blurred_level1);
      fFreeHandle(blurred_level2);

      cudaEndPhase(timer, phasei++, false);
      //printf("freed stuff cpu\n");
      timer = cudaStartPhase();

      blurredImage = imageBlur(Ic);
      /** Image pyramid **/
      blurred_level1 = blurredImage;
      blurred_level2 = imageResize(blurredImage);

      /** Gradient image computation, for all scales **/
      verticalEdge_level1 = calcSobel_dX(blurred_level1);   
      horizontalEdge_level1 = calcSobel_dY(blurred_level1); 
      
      verticalEdge_level2 = calcSobel_dX(blurred_level2); 
      horizontalEdge_level2 = calcSobel_dY(blurred_level2);

      cudaEndPhase(timer, phasei++, true);
      //printf("blur,resize,sobel cpu\n");
      timer = cudaStartPhase();
    }
    if(use_gpu)
    {
      iCopyToGPU(Ic, d_Ic);
      d_previousFrameBlurred_level1 = d_blurred_level1;
      d_previousFrameBlurred_level2 = d_blurred_level2;
      d_blurredImage = fMallocCudaArray(rows, cols);
      d_blurred_level2 = fMallocCudaArray(resizeRows, resizeCols);
      d_horizontalEdge_level2 = fMallocCudaArray(resizeRows, resizeCols);
      d_verticalEdge_level2 = fMallocCudaArray(resizeRows, resizeCols);
      d_edgeTemp_level2 = fMallocCudaArray(resizeRows, resizeCols);
      d_blurred_level1 = d_blurredImage;

      d_horizontalEdge_level1 = d_horizontalEdgeImage;//fMallocCudaArray(rows, cols);
      d_verticalEdge_level1 = d_verticalEdgeImage;//fMallocCudaArray(rows, cols);

      cudaEndPhase(timer, phasei++, false);
      //printf("malloc/copy stuff gpu\n");
      timer = cudaStartPhase();

      blockdim.x = 64; blockdim.y = 4;
      griddim.x = cols / blockdim.x + (cols % blockdim.x == 0 ? 0 : 1);
      griddim.y = rows / blockdim.y + (rows % blockdim.y == 0 ? 0 : 1);
      imageBlurHorizontal_kernel<<<griddim, blockdim>>>(d_Ic, d_tempImage);
      imageBlurVertical_kernel<<<griddim, blockdim>>>(d_tempImage, d_blurredImage);
      GPUERRCHK;


      d_blurred_level1 = d_blurredImage;
      resizeRows = floor((rows+1)/2);
      resizeCols = floor((cols+1)/2);
    //F2D* d_blurred_temp, *d_blurred_level2;
      griddim.x = resizeCols / blockdim.x + (resizeCols % blockdim.x == 0 ? 0 : 1);
      //printf("launching imageResizeHorizontal_kernel with grid %dx%dx%d and tb %dx%dx%d\n", griddim.x, griddim.y, griddim.z, blockdim.x, blockdim.y, blockdim.z);
      imageResizeHorizontal_kernel<<<griddim, blockdim>>>(d_blurred_level1, d_blurred_temp);
      griddim.y = resizeRows / blockdim.y + (resizeRows % blockdim.y == 0 ? 0 : 1);
      //printf("launching imageResizeVertical_kernel with grid %dx%dx%d and tb %dx%dx%d\n", griddim.x, griddim.y, griddim.z, blockdim.x, blockdim.y, blockdim.z);
      imageResizeVertical_kernel<<<griddim, blockdim>>>(d_blurred_temp, d_blurred_level2);

      /* sobel filtering */
      blockdim.x = 64; blockdim.y = 4;
      griddim.y = rows / blockdim.y + (rows % blockdim.y == 0 ? 0 : 1);
      griddim.x = cols / blockdim.x + (cols % blockdim.x == 0 ? 0 : 1);
      //printf("launching calcSobelHor with grid %dx%dx%d and tb %dx%dx%d\n", griddim.x, griddim.y, griddim.z, blockdim.x, blockdim.y, blockdim.z);
      calcSobelHorizontal_dX_kernel<<<griddim, blockdim>>>(d_blurred_level1, d_edgeTemp);
      calcSobelVertical_dX_kernel<<<griddim, blockdim>>>(d_edgeTemp, d_verticalEdge_level1);

      calcSobelVertical_dY_kernel<<<griddim, blockdim>>>(d_blurred_level1, d_edgeTemp);
      calcSobelHorizontal_dY_kernel<<<griddim, blockdim>>>(d_edgeTemp, d_horizontalEdge_level1);

      /* level 2 */
      griddim.y = resizeRows / blockdim.y + (resizeRows % blockdim.y == 0 ? 0 : 1);
      griddim.x = resizeCols / blockdim.x + (resizeCols % blockdim.x == 0 ? 0 : 1);
      calcSobelHorizontal_dX_kernel<<<griddim, blockdim>>>(d_blurred_level2, d_edgeTemp_level2);
      calcSobelVertical_dX_kernel<<<griddim, blockdim>>>(d_edgeTemp_level2, d_verticalEdge_level2);

      calcSobelVertical_dY_kernel<<<griddim, blockdim>>>(d_blurred_level2, d_edgeTemp_level2);
      calcSobelHorizontal_dY_kernel<<<griddim, blockdim>>>(d_edgeTemp_level2, d_horizontalEdge_level2);

      cudaEndPhase(timer, phasei++, true);
      //printf("blur, resize,sobel gpu\n");
      timer = cudaStartPhase();

      blurred_level1 = fMallocHandle(rows, cols);
      previousFrameBlurred_level1 = fMallocHandle(rows, cols);
      verticalEdge_level1 = fMallocHandle(rows, cols);
      horizontalEdge_level1 = fMallocHandle(rows, cols);

      blurred_level2 = fMallocHandle(resizeRows, resizeCols);
      previousFrameBlurred_level2 = fMallocHandle(resizeRows, resizeCols);
      verticalEdge_level2 = fMallocHandle(resizeRows, resizeCols);
      horizontalEdge_level2 = fMallocHandle(resizeRows, resizeCols);

      fCopyFromGPU(blurred_level1, d_blurred_level1);
      fCopyFromGPU(previousFrameBlurred_level1, d_previousFrameBlurred_level1);
      fCopyFromGPU(verticalEdge_level1, d_verticalEdge_level1);
      fCopyFromGPU(horizontalEdge_level1, d_horizontalEdge_level1);
      fCopyFromGPU(blurred_level2, d_blurred_level2);
      fCopyFromGPU(previousFrameBlurred_level2, d_previousFrameBlurred_level2);
      fCopyFromGPU(verticalEdge_level2, d_verticalEdge_level2);
      fCopyFromGPU(horizontalEdge_level2, d_horizontalEdge_level2);

      cudaEndPhase(timer, phasei++, false);
      //printf("copy back gpu\n");
      timer = cudaStartPhase();
    }
    
    newpoints = fSetArray(2, features->width, 0);
        
    /** Based on features computed in the previous frame, find correspondence in the current frame. "status" returns the index of corresponding features **/
    status = calcPyrLKTrack(previousFrameBlurred_level1, previousFrameBlurred_level2, verticalEdge_level1, verticalEdge_level2, horizontalEdge_level1, horizontalEdge_level2, blurred_level1, blurred_level2, features, features->width, WINSZ, accuracy, LK_ITER, newpoints);
    
    fFreeHandle(verticalEdge_level1);
    fFreeHandle(verticalEdge_level2);
    fFreeHandle(horizontalEdge_level1);
    fFreeHandle(horizontalEdge_level2);
    fFreeHandle(previousFrameBlurred_level1);
    fFreeHandle(previousFrameBlurred_level2);

    if(use_gpu)
    {
      cudaFree(d_verticalEdge_level2);
      cudaFree(d_horizontalEdge_level2);
      cudaFree(d_edgeTemp_level2);
      cudaFree(previousFrameBlurred_level1);
      cudaFree(previousFrameBlurred_level2);
    }
    
    /** Populate newpoints with features that had correspondence with previous frame features **/
    np_temp = fDeepCopy(newpoints);
    if(status->width > 0 )
    {
        k = 0;
        numFind=0;
        for(i=0; i<status->width; i++)
        {
            if( asubsref(status,i) == 1)
                numFind++;
        }
        fFreeHandle(newpoints);
        newpoints = fSetArray(2, numFind, 0);

        for(i=0; i<status->width; i++)
        {
            if( asubsref(status,i) == 1)
            {
                subsref(newpoints,0,k) = subsref(np_temp,0,i);
                subsref(newpoints,1,k++) = subsref(np_temp,1,i);
            }
        }    
    }    
        
    iFreeHandle(status);
    iFreeHandle(Ic);
    fFreeHandle(np_temp);
    fFreeHandle(features);
    /** Populate newpoints into features **/
    features = fDeepCopy(newpoints);
    fFreeHandle(newpoints);
    
    /* Timing utils */
    end = photonEndTiming();
    elt = photonReportTiming(start, end);
    elapsed[0] += elt[0];
    elapsed[1] += elt[1];
    
    free(start);
    free(elt);
    free(end);   
    cudaEndPhase(timer, phasei++, true);
    //printf("a bunch of cpu stuff\n");
    timer = cudaStartPhase();
}

    if(use_gpu)
    {
      cudaFree(d_Ic);
      cudaFree(d_tempImage);
      cudaFree(d_blurredImage);
      cudaFree(d_blurred_temp);
      cudaFree(d_blurred_level2);
      cudaFree(d_verticalEdgeImage);
      cudaFree(d_edgeTemp);
      cudaFree(d_horizontalEdgeImage);
    }
    //end here
    cudaEndPhase(timer_start, 0, true);
#ifdef CHECK   
    /* Self checking */
    {
        int ret=0;
        float tol = 2.0;
#ifdef GENERATE_OUTPUT
        fWriteMatrix(features, argv[1]);
#endif
        ret = fSelfCheck(features, argv[1], tol); 
        if (ret == -1)
            printf("Error in Tracking Map\n");
    }
#endif
    
    photonPrintTiming(elapsed);

    fFreeHandle(blurred_level1);
    fFreeHandle(blurred_level2);
    fFreeHandle(features);

    free(elapsed);

    return 0;

}



