/***********************************************************************************
 *  Example cuda kernel file
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

void kernelWrapper();

__global__ void Kernel( int i)
{
  //do stuff here
}


void kernelWrapper(bool use_gpu, bool gpu_transfer ) 
{
	// setup execution parameters
	dim3  grid( 1, 1, 1);
	dim3  threads( 128, 1, 1);

  Kernel<<< grid, threads, 0 >>>(0);

}
