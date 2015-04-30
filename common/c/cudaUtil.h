#ifndef _CUDA_UTIL_

#define _CUDA_UTIL_

#include <assert.h>

// Error checking
#ifdef DEBUG
#define GPUERRCHK { gpuAssert((cudaGetLastError()), __FILE__, __LINE__); }
#else
#define GPUERRCHK
#endif

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   //printf("gpuAsserting\n");
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline void printSome(F2D* array)
{
  for(int i=0; i<10; i++) {
    printf("%f, ", asubsref(array, i));
  }
  printf("\n");
}

inline void printSome(I2D* array)
{
  for(int i=0; i<10; i++) {
    printf("%d, ", asubsref(array, i));
  }
  printf("\n");
}

inline void compareArrays(F2D* array1, F2D* array2)
{
  assert(array1->height==array2->height && array1->width==array2->width);
  for(int y=0; y<array1->height; y++)
  {
    for(int x=0; x<array1->width; x++)
    {
      float v1 = subsref(array1,y,x);
      float v2 = subsref(array2,y,x);
      if(v1!=v2)
      {
        printf("mismatch at %d,%d: %f != %f\n", x, y, v1, v2);
        return;
      }
    }
  }
}

inline void compareArrays(I2D* array1, I2D* array2)
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
}

//transfer stuff
F2D* fMallocCudaArray(int nrows, int ncols);
F2D* fMallocCudaArray(F2D* copy);
cudaError_t fCopyToGPU(F2D* host, F2D* device);
cudaError_t fCopyFromGPU(F2D* host, F2D* device);
F2D* fMallocAndCopy(F2D* host_array);
cudaError_t fCopyAndFree(F2D* host_array, F2D* device_array);
I2D* iMallocCudaArray(int nrows, int ncols);
I2D* iMallocCudaArray(I2D* copy);
cudaError_t iCopyToGPU(I2D* host, I2D* device);
cudaError_t iCopyFromGPU(I2D* host, I2D* device);
I2D*  iMallocAndCopy(I2D* host_array);
cudaError_t iCopyAndFree(I2D* host_array, I2D* device_array);

//timing stuff
unsigned int* cudaStartTransfer();
void cudaEndTransfer(unsigned int* start);
unsigned int* cudaStartPhase();
unsigned int cudaEndPhase(unsigned int* start, int phase);

#endif
