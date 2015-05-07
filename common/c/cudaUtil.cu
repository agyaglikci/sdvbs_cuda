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

F2D* fMallocCudaArray(int nrows, int ncols, bool set_dimensions)
{
  //int out;
  F2D* out;
  cudaMalloc( (void**) &out, sizeof(F2D)+sizeof(float)*nrows*ncols);
  if(set_dimensions) {
    //just copy row and col dimensions
    F2D copy_dimensions;
    copy_dimensions.width = ncols;
    copy_dimensions.height = nrows;
    cudaError_t err = cudaMemcpy( out, &copy_dimensions, sizeof(F2D), cudaMemcpyHostToDevice);
  }
  //GPUERRCHK;
  return out;
}

F2D* fMallocCudaArray(F2D* copy)
{
  int nrows = copy->height;
  int ncols = copy->width;
  //printf("fallocating h:%d w:%d\n", nrows, ncols);
  F2D* out = fMallocCudaArray(nrows, ncols, true);
  return out;
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
  //printf("copying %d bytes from 0x%x to 0x%x\n", sizeof(F2D)+sizeof(float)*rows*cols, device, host);
  GPUERRCHK; 
  cudaError_t ret = cudaMemcpy( host, device, sizeof(F2D)+sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);
  //GPUERRCHK; 
  return ret;
}

F2D* fMallocAndCopy(F2D* host_array)
{
  int rows = host_array->height;
  int cols = host_array->width;
  GPUERRCHK;
  F2D* device_array = fMallocCudaArray(rows, cols, false);
  GPUERRCHK;
  fCopyToGPU(host_array, device_array);
  //if(cudaSuccess != fCopyToGPU(host_array, device_array)) {
  //  return 0;
  //}
  return device_array;
}

cudaError_t fCopyAndFree(F2D* host_array, F2D* device_array)
{
  fCopyFromGPU(host_array, device_array);
  return cudaFree(device_array);
}

I2D* iMallocCudaArray(int nrows, int ncols, bool set_dimensions)
{
  I2D* out;
  //int out;
  cudaMalloc( (void**) &out, sizeof(I2D)+sizeof(int)*nrows*ncols);
  if(set_dimensions) {
    //just copy row and col dimensions
    I2D copy_dimensions;
    copy_dimensions.width = ncols;
    copy_dimensions.height = nrows;
    cudaError_t err = cudaMemcpy( out, &copy_dimensions, sizeof(I2D), cudaMemcpyHostToDevice);
  }
  GPUERRCHK;
  return out;
  //return (I2D*) ((void*)(out));
}

I2D* iMallocCudaArray(I2D* copy)
{
  int nrows = copy->height;
  int ncols = copy->width;
  //printf("iallocating h:%d w:%d\n", nrows, ncols);
  I2D* out = iMallocCudaArray(nrows, ncols, true);
  return out;
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
  int numbytes = sizeof(I2D)+sizeof(int)*rows*cols;
  //printf("CopyFromGPU h:0x%x d:0x%x numbytes:%d\n", host, device, numbytes);
  return cudaMemcpy( host, device, numbytes, cudaMemcpyDeviceToHost);
}

I2D*  iMallocAndCopy(I2D* host_array)
{
  int rows = host_array->height;
  int cols = host_array->width;
  I2D* device_array = iMallocCudaArray(rows, cols, false);
  iCopyToGPU(host_array, device_array);
  GPUERRCHK;
  //if(cudaSuccess != iCopyToGPU(host_array, device_array)) {
  //  assert(0);
  //}
  return device_array;
}

cudaError_t iCopyAndFree(I2D* host_array, I2D* device_array)
{
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

void cudaEndPhase(unsigned int* start, int phase, bool is_compute, bool free_start)
{
  unsigned int* end=photonEndTiming();
	unsigned long long starttime = (((unsigned long long)0x0) | start[1]) << 32 | start[0];
	unsigned long long endtime = (((unsigned long long)0x0) | end[1]) << 32 | end[0];
	unsigned long long diff = endtime - starttime;
  printf("Phase %d, %d, %d, %lu\n", phase, is_compute, !is_compute, diff);

  //unsigned int* elapsed=photonReportTiming(start, end);
  //if(elapsed[1] == 0)
  //  printf("Phase %d cycles\t\t- %u\n\n", phase, elapsed[0]);
  //else
  //  printf("Phase %d cycles\t\t- %u,%u\n\n", phase, elapsed[1], elapsed[0]);
  if(free_start)
  {
    free(start);
  }
  free(end);
}

void printSome(F2D* array)
{
  for(int i=0; i<25; i++) {
    //printf("%f, ", asubsref(array, i));
    for(int j=0; j<13; j++) {
      printf("%f ", subsref(array, i, j));
    }
    printf("\n");
  }
  printf("\n");
}

void printSome(I2D* array) 
{
  for(int i=0; i<10; i++) {
    printf("%d, ", asubsref(array, i));
  }
  printf("\n");
}

void printSomeCuda(F2D* array, int rows, int cols) 
{
  F2D* host_copy = fMallocHandle(rows, cols);
  fCopyFromGPU(host_copy, array);
  printSome(host_copy);
}

void compareArrays(F2D* array1, F2D* array2)
{
  //printf("comparing %dx%d 0x%x and %dx%d 0x%x\n", array1->height, array1->width, array1, array2->height, array2->width, array2);
  if(array1->height!=array2->height || array1->width!=array2->width)
  {
    printf("compareArrays error: h %d!=%d or w %d!=%d\n", array1->height, array2->height, array1->width, array2->width);
    assert(0);
  }
  for(int y=0; y<array1->height; y++)
  {
    for(int x=0; x<array1->width; x++)
    {
      float v1 = subsref(array1,y,x);
      float v2 = subsref(array2,y,x);
      float diff = v1 - v2;
      //if(diff < -0.0000001 || diff > 0.0000001) 
      if(diff < -1 || diff > 1) 
      {
        printf("mismatch at %d,%d: %f != %f\n", x, y, v1, v2);
        return;
      }
    }
  }
  printf("No mismatch\n");
}

void compareArraysCuda(F2D* host, F2D* device)
{
  int rows = host->height;
  int cols = host->width;
  F2D* host_copy = fMallocHandle(rows, cols);
  fCopyFromGPU(host_copy, device);
  compareArrays(host_copy, host);
}

void compareArrays(I2D* array1, I2D* array2)
{
  assert(array1->height==array2->height && array1->width==array2->width);
  for(int y=0; y<array1->height; y++)
  {
    for(int x=0; x<array1->width; x++)
    {
      int v1 = subsref(array1,y,x);
      int v2 = subsref(array2,y,x);
      if(v1!=v2) 
      {
        printf("mismatch at %d,%d: %d != %d\n", x, y, v1, v2);
        return;
      }
    }
  }
  printf("No mismatch\n");
}
