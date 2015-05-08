/********************************
Author: Sravanthi Kota Venkata
********************************/

#include <math.h>
#include "sift.h"
#include <time.h>
#include <sys/time.h>

void normalizeImage(F2D* image)
{
    int i;
    int rows;
    int cols;
    int tempMin = 10000, tempMax = -1;
    rows = image->height;
    cols = image->width;

    for(i=0; i<(rows*cols); i++)
        if(tempMin > asubsref(image,i))
            tempMin = asubsref(image,i);

    for(i=0; i<(rows*cols); i++)
        asubsref(image,i) = asubsref(image,i) - tempMin;

    for(i=0; i<(rows*cols); i++)
        if(tempMax < asubsref(image,i))
            tempMax = asubsref(image,i);

    for(i=0; i<(rows*cols); i++)
        asubsref(image,i) = ( asubsref(image,i) / (tempMax+0.0) );
}

/** SIFT- Scale Invariant Feature Transform. This algorithm is based on
    David Lowe's implementation of sift. So, we will use the parameter
    values from Lowe's implementation.
    See: http://www.cs.ubc.ca/~lowe/keypoints/

    SIFT extracts from an image a collection of frames or keypoints. These
    are oriented disks attacked to blob-like structures of the image. As
    the image translates, rotates and scales, the frames track these blobs
    and the deformation.

    'BoundaryPoint' [bool]
    If set to 1, frames too close to the image boundaries are discarded.

    'Sigma0' [pixels]
    Smoothing of the level 0 of octave 0 of the scale space.
    (Note that Lowe's 1.6 value refers to the level -1 of octave 0.)

    'SigmaN' [pixels]
    Nominal smoothing of the input image. Typically set to 0.5.

    'Threshold'
    Threshold used to eliminate weak keypoints. Typical values for
    intensity images in the range [0,1] are around 0.01. Smaller
    values mean more keypoints.

**/

F2D* sift(F2D* I, int use_gpu, int gpu_transfer)
{
    printf("%s: %d\n", __FILE__, __LINE__);
    int rows, cols;
    int subLevels, omin, Octaves, r, smin, smax, intervals, o;
    float sigman, sigma0, thresh;
    int discardBoundaryPoints;
    F2D **gss, *tfr;
    F2D **dogss;
    I2D* i_, *s_;
    F2D *oframes, *frames;
    int firstIn=1;
    float minVal;
    I2D* tx1, *ty1, *ts1;
    I2D* x1, *y1, *s1, *txys;

    rows = I->height;
    cols = I->width;

    // device copies
    F2D * d_I;

    /**
    Lowe's choices
    Octaves - octave
    subLevels - sub-level for image
    sigma - sigma value for gaussian kernel, for smoothing the image
    At each successive octave, the data is spatially downsampled by half
    **/

    subLevels = 3;
    omin = -1;
    minVal = log2f(MIN(rows,cols));
    Octaves = (int)(floor(minVal))-omin-4;   /* Upto 16x16 images */
    sigma0 = 1.6 * pow(2, (1.0/subLevels));
    sigman = 0.5;
    thresh = (0.04/subLevels)/2;
    r = 10;

#ifdef test
    subLevels = 1;
    Octaves = 1;
    sigma0 = pow(0.6 * 2, 1);
    sigman = 1.0;
    thresh = (1/subLevels)/2;
    r = 1;
#endif

    discardBoundaryPoints = 1 ;

    smin = -1;
    smax = subLevels+1;
    intervals = smax - smin + 1;


    /** Normalize the input image to lie between 0-1 **/
    normalizeImage(I);

    /**
        We build gaussian pyramid for the input image. Given image I,
        we sub-sample the image into octave 'Octaves' number of levels. At
        each level, we smooth the image with varying sigman values.

        Gaussiam pyramid can be assumed as a 2-D matrix, where each
        element is an image. Number of rows corresponds to the number
        of scales of the pyramid (octaves, "Octaves"). Row 0 (scale 0) is the
        size of the actual image, Row 1 (scale 1) is half the actual
        size and so on.

        At each scale, the image is smoothened with different sigma values.
        So, each row has "intervals" number of smoothened images, starting
        with least blurred.

        gss holds the entire gaussian pyramid.
    **/
    struct timespec start, end;
    //struct timeval start1, end1;
    //gettimeofday(&start1, NULL);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    printf("%s: %d\n", __FILE__, __LINE__);
    if (use_gpu)
    {
        gss = cuda_gaussianss(I, sigman, Octaves, subLevels, omin, smin, smax, sigma0);
        dogss = gss;
        //dogss = diffss(gss, Octaves, intervals, gpu_transfer, 0);

    }
    else
    {
        gss = gaussianss(I, sigman, Octaves, subLevels, omin, -1, subLevels+1, sigma0, use_gpu, gpu_transfer);
        dogss = diffss(gss, Octaves, intervals, gpu_transfer, 0);
    }


    //printf("%s: %d\n", __FILE__, __LINE__);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    //gettimeofday(&end1, NULL);


    printf("gaussians\n");
    printf("Clock cycles: %llu\n", (long long unsigned int) 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec);
    //printf("Clock cycles: %llu\n", (long long unsigned int) 1000000000L * (end1.tv_sec - start1.tv_sec) + 1000 * (end1.tv_usec - start1.tv_usec));


    /**
        Once we build the gaussian pyramid, we compute DOG, the
        Difference of Gaussians. At every scale, we do:

        dogss[fixedScale][0] = gss[fixedScale][1] - gss[fixedScale][0]

        Difference of gaussian gives us edge information, at each scale.
        In order to detect keypoints on the intervals per octave, we
        inspect DOG images at highest and lowest scales of the octave, for
        extrema detection.
    **/
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    printf("diffs\n");
    printf("Clock cycles: %llu\n", (long long unsigned int) 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    for(o=0; o<Octaves; o++)
    {
        F2D *temp;
        F2D *t;
        F2D *negate;
        F2D *idxf, *idxft;
        I2D *idx;
        int i1;
        I2D *x, *y;
        int sizeRows = dogss[o*intervals]->height;
        int sizeCols = dogss[o*intervals]->width;
        struct timespec start1, end1;

        {
            int i,j,k=0;
            temp = fMallocHandle(intervals-1, sizeRows*sizeCols);
            negate = fMallocHandle(intervals-1, sizeRows*sizeCols);

            /**
                Keypoints are detected as points of local extrema of the
                DOG pyramid, at a given octave. In softlocalmax function, the
                keypoints are extracted by looking at 9x9x9 neighborhood samples.

                We populate temp and negate arrays with the values of the DOG
                pyramid for a given octave, o. Since we are interested in both
                local maxima and minima, we compute negate matrix, which is the
                negated values of the DOG pyramid.

            **/

            for(i1=0; i1<(intervals-1); i1++)
            {
                for(j=0; j<sizeCols; j++)
                {
                    for(i=0; i<sizeRows; i++)
                    {
                        asubsref(temp,k) = subsref(dogss[o*intervals+i1],i,j);
                        asubsref(negate,k++) = -subsref(dogss[o*intervals+i1],i,j);
                    }
                }
            }
        }

        /**
            siftlocalmax returns indices k, that correspond to local maxima and
            minima.
	        The 80% tricks discards early very weak points before refinement.
        **/
        idxf = siftlocalmax( temp, 0.8*thresh, intervals, sizeRows, sizeCols);
        t = siftlocalmax( negate, 0.8*thresh, intervals, sizeRows, sizeCols);

        idxft = fHorzcat(idxf, t);

        /**
            Since indices is the 1-D index of the temp/negate arrays, we compute
            the x,y and intervals(s) co-ordinates corresponding to each index.
        **/

        idx = ifDeepCopy(idxft);


        x = iSetArray(idx->height,idx->width,0);
        y = iSetArray(idx->height,idx->width,0);
        s_ = iSetArray(idx->height,idx->width,0);

        {
            int i, j;
            for(i=0; i<idx->height; i++)
             {
                 for(j=0; j<idx->width; j++)
                 {
                    int v, u, w, z;
                    w = subsref(idx,i,j);
                    v = ceil((w/(sizeRows*sizeCols)) + 0.5);
                    u = floor(w/(sizeRows*sizeCols));
                    z = w - (sizeRows*sizeCols*u);

                    /** v is the interval number, s **/
                    subsref(s_,i,j) = v;
                    /** row number of the index **/
                    subsref(y,i,j) = ceil((z / sizeRows)+0.5);
                    /** col number of the index **/
                    subsref(x,i,j) = z - (sizeCols * floor(z / sizeRows));
                 }
             }
        }

        {

            tx1 = isMinus(x, 1);
            ty1 = isMinus(y, 1);
            ts1 = isPlus(s_, (smin-1));

            x1 = iReshape(tx1, 1, (tx1->height*tx1->width));
            y1 = iReshape(ty1, 1, (ty1->height*ty1->width));
            s1 = iReshape(ts1, 1, (ts1->height*ts1->width));

            txys = iVertcat(y1, x1);
            i_ = iVertcat(txys, s1);

        }
        /**
            Stack all x,y,s into oframes.
            Row 0 of oframes = x
            Row 1 of oframes = y
            Row 2 of oframes = s
        **/
        oframes = fiDeepCopy(i_);

        {
            F2D* temp;
            temp = oframes;

            /**
                Remove points too close to the boundary
            **/

            if(discardBoundaryPoints)
                oframes = filterBoundaryPoints(sizeRows, sizeCols, temp);
            fFreeHandle(temp);
        }

        /**
            Refine the location, threshold strength and remove points on edges
        **/
        if( asubsref(oframes,0) != 0)
        {
            F2D* temp_;
            temp_ = fTranspose(oframes);
            fFreeHandle(oframes);
            oframes = siftrefinemx(temp_, temp, smin, thresh, r, sizeRows, sizeCols, intervals-1);
            fFreeHandle(temp_);

            if( firstIn == 0)
            {
                tfr = fDeepCopy(frames);
                fFreeHandle(frames);
                frames = fHorzcat(tfr, oframes);
                fFreeHandle(tfr);
            }
            else
                frames = fDeepCopy(oframes);
            firstIn = 0;
        }
        else if(Octaves == 1)
            frames = fDeepCopy(oframes);

        fFreeHandle(oframes);
        iFreeHandle(y);
        iFreeHandle(x);
        iFreeHandle(s_);
        iFreeHandle(y1);
        iFreeHandle(x1);
        iFreeHandle(s1);
        iFreeHandle(ty1);
        iFreeHandle(tx1);
        iFreeHandle(ts1);
        iFreeHandle(txys);
        iFreeHandle(i_);
        iFreeHandle(idx);
        fFreeHandle(idxf);
        fFreeHandle(idxft);
        fFreeHandle(temp);
        fFreeHandle(t);
        fFreeHandle(negate);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    printf("extract\n");
    printf("Clock cycles: %llu\n", (long long unsigned int) 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec);


    { int s;
    if (use_gpu == 0)
    {
        for(o=0; o<Octaves; o++)
        {
            for(s=0; s<(intervals-1); s++)
            {
                fFreeHandle(dogss[o*intervals+s]);
            }
        }
        free(dogss);
    }


    for(o=0; o<Octaves; o++)
    {
        for(s=0; s<(intervals); s++)
        {
            fFreeHandle(gss[o*intervals+s]);
        }
    }
    }
    free(gss);

    return frames;
}



