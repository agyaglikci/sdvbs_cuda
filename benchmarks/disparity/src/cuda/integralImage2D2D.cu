/********************************
Author: Sravanthi Kota Venkata
********************************/

#include <stdio.h>
#include <stdlib.h>
#include "disparity.h"

void integralImage2D2D(F2D* SAD, F2D* integralImg)
{
    int nr, nc, i, j;
    
    nr = SAD->height;
    nc = SAD->width;
    
    for(i=0; i<nc; i++)
        subsref(integralImg,0,i) = subsref(SAD,0,i);
    
    for(i=1; i<nr; i++)
        for(j=0; j<nc; j++)
        {
            subsref(integralImg,i,j) = subsref(integralImg, (i-1), j) + subsref(SAD,i,j);
        }
//#ifdef DEBUG
//    printf("gpu integralImg1:\n");
//    for(int el=0; el<10; el++) 
//    {
//      printf("%f, ", subsref(integralImg, el, el));
//    }
//    printf("\n");
//#endif

    for(i=0; i<nr; i++)
        for(j=1; j<nc; j++)
            subsref(integralImg,i,j) = subsref(integralImg, i, (j-1)) + subsref(integralImg,i,j);
//#ifdef DEBUG
//    printf("cpu integralImg2:\n");
//    for(int el=300; el<600; el++) 
//    {
//      //printf("%f, ", subsref(integralImg, el, el));
//      printf("%f, ", asubsref(integralImg, el));
//    }
//    printf("\n");
//#endif

    return;
    
}
