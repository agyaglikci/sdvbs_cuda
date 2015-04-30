/***********************************************************************************
 *  Example cuda kernel file
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

extern "C" {
#include "sdvbs_common.h"
}
#include "cudaUtil.h"

void kernelWrapper(bool use_gpu, bool gpu_transfer);

F2D* fMallocCudaArray(int nrows, int ncols)
{
  //int out;
  F2D* out;
  cudaMalloc( (void**) &out, sizeof(F2D)+sizeof(float)*nrows*ncols);
  //printf("fallocating %d bytes\n", sizeof(F2D)+sizeof(float)*nrows*ncols);
  GPUERRCHK;
  return out;
}

F2D* fMallocCudaArray(F2D* copy)
{
  int nrows = copy->height;
  int ncols = copy->width;
  //printf("fallocating h:%d w:%d\n", nrows, ncols);
  return fMallocCudaArray(nrows, ncols);
}

cudaError_t fCopyToGPU(F2D* host, F2D* device)
{
  int rows = host->height;
  int cols = host->width;
  return cudaMemcpy( device, host, sizeof(F2D)+sizeof(float)*rows*cols, cudaMemcpyHostToDevice);
}

cudaError_t fCopyFromGPU(F2D* host, F2D* device)
{
  int rows = host->height;
  int cols = host->width;
  //printf("copying %d bytes\n", sizeof(F2D)+sizeof(float)*rows*cols);
  return cudaMemcpy( host, device, sizeof(F2D)+sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);
}

F2D* fMallocAndCopy(F2D* host_array)
{
  int rows = host_array->height;
  int cols = host_array->width;
  F2D* device_array = fMallocCudaArray(rows, cols);
  fCopyToGPU(host_array, device_array);
  GPUERRCHK;
  //if(cudaSuccess != fCopyToGPU(host_array, device_array)) {
  //  return 0;
  //}
  return device_array;
}

cudaError_t fCopyAndFree(F2D* host_array, F2D* device_array)
{
  //int rows = host_array->height;
  //int cols = host_array->width;
  fCopyFromGPU(host_array, device_array);
  return cudaFree(device_array);
}

I2D* iMallocCudaArray(int nrows, int ncols)
{
  I2D* out;
  //int out;
  cudaMalloc( (void**) &out, sizeof(I2D)+sizeof(int)*nrows*ncols);
  //printf("iallocating %d bytes\n", sizeof(F2D)+sizeof(float)*nrows*ncols);
  GPUERRCHK;
  //printf("iallocated h:%d w:%d\n", nrows, ncols);
  return out;
  //return (I2D*) ((void*)(out));
}

I2D* iMallocCudaArray(I2D* copy)
{
  int nrows = copy->height;
  int ncols = copy->width;
  //printf("iallocating h:%d w:%d\n", nrows, ncols);
  return iMallocCudaArray(nrows, ncols);
  //printf("iallocated h:%d w:%d\n", nrows, ncols);
}

cudaError_t iCopyToGPU(I2D* host, I2D* device)
{
  //printf("icopying h:0x%x d:0x%x\n", host, device);
  int rows = host->height;
  int cols = host->width;
  return cudaMemcpy(device, host, sizeof(I2D)+sizeof(int)*rows*cols, cudaMemcpyHostToDevice);
}

cudaError_t iCopyFromGPU(I2D* host, I2D* device)
{
  int rows = host->height;
  int cols = host->width;
  //int numbytes = sizeof(I2D)+sizeof(int)*rows*cols;
  //printf("CopyFromGPU h:0x%x d:0x%x numbytes:%d\n", host, device, numbytes);
  return cudaMemcpy( host, device, sizeof(I2D)+sizeof(int)*rows*cols, cudaMemcpyDeviceToHost);
}

I2D*  iMallocAndCopy(I2D* host_array)
{
  I2D* device_array = iMallocCudaArray(host_array);
  iCopyToGPU(host_array, device_array);
  GPUERRCHK;
  //if(cudaSuccess != iCopyToGPU(host_array, device_array)) {
  //  assert(0);
  //}
  return device_array;
}

cudaError_t iCopyAndFree(I2D* host_array, I2D* device_array)
{
  //int rows = host_array->height;
  //int cols = host_array->width;
  iCopyFromGPU(host_array, device_array);
  return cudaFree(device_array);
}

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

unsigned int* cudaStartTransfer()
{
  unsigned int* start=photonStartTiming();
  return start;
}

unsigned int* cudaStartPhase()
{
  unsigned int* start=photonStartTiming();
  return start;
}

unsigned int cudaEndPhase(unsigned int* start, int phase)
{
  unsigned int* end=photonEndTiming();
  unsigned int* elapsed=photonReportTiming(start, end);
  if(elapsed[1] == 0)
    printf("Phase %d cycles\t\t- %u\n\n", phase, elapsed[0]);
  else
    printf("Phase %d cycles\t\t- %u%u\n\n", phase, elapsed[0], elapsed[1]);
  free(start);
  free(end);
  return 0;
}

