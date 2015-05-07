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

//transfer stuff
F2D* fMallocCudaArray(int nrows, int ncols, bool set_dimensions=true);
F2D* fMallocCudaArray(F2D* copy);
cudaError_t fCopyToGPU(F2D* host, F2D* device);
cudaError_t fCopyFromGPU(F2D* host, F2D* device);
F2D* fMallocAndCopy(F2D* host_array);
cudaError_t fCopyAndFree(F2D* host_array, F2D* device_array);
I2D* iMallocCudaArray(int nrows, int ncols, bool set_dimensions=true);
I2D* iMallocCudaArray(I2D* copy);
cudaError_t iCopyToGPU(I2D* host, I2D* device);
cudaError_t iCopyFromGPU(I2D* host, I2D* device);
I2D*  iMallocAndCopy(I2D* host_array);
cudaError_t iCopyAndFree(I2D* host_array, I2D* device_array);

//timing stuff
unsigned int* cudaStartTransfer();
void cudaEndTransfer(unsigned int* start);
unsigned int* cudaStartPhase();
void cudaEndPhase(unsigned int* start, int phasei=-1, bool is_compute=true, bool free_start=true);


//print stuff, other stuff

void printSome(F2D* array);

void printSome(I2D* array) ;

void printSomeCuda(F2D* array, int rows, int cols) ;

void compareArrays(F2D* array1, F2D* array2);
void compareArraysCuda(F2D* array1, F2D* array2);

void compareArrays(I2D* array1, I2D* array2);
#endif
