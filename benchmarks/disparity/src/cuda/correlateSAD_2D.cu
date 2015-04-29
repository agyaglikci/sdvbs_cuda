/********************************
Author: Sravanthi Kota Venkata
********************************/

#include <stdio.h>
#include <stdlib.h>
#include "disparity.h"

void correlateSAD_2D(I2D* Ileft, I2D* Iright, I2D* Iright_moved, int win_sz, int disparity, F2D* SAD, F2D* integralImg, F2D* retSAD)
{
    int rows, cols;
    int i, j, endRM;
    I2D *range;
    
    range = iMallocHandle(1,2);
    subsref(range,0,0) = 0;
    subsref(range,0,1) = disparity;

    rows = Iright_moved->height;
    cols = Iright_moved->width;

    for(i=0; i<rows*cols; i++)
        asubsref(Iright_moved,i) = 0;
    
    padarray4(Iright, range, -1, Iright_moved);
//#ifdef DEBUG
//    printf("cpu padarray:\n");
//    printSome(Iright_moved);
//#endif
//#ifdef DEBUG
//    printf("cpu Ileft:\n");
//    printSome(Ileft);
//#endif
   
    computeSAD(Ileft, Iright_moved, SAD);
//#ifdef DEBUG
//    printf("cpu computeSAD:\n");
//    printSome(SAD);
//#endif
    integralImage2D2D(SAD, integralImg);
//#ifdef DEBUG
//    printf("cpu integralImg:\n");
//    printSome(integralImg);
//#endif
    finalSAD(integralImg, win_sz, retSAD);

    iFreeHandle(range);    
    return;
}

