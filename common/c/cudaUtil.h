#ifndef _CUDA_UTIL_

#define _CUDA_UTIL_

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
      //fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//transfer stuff
F2D* fMallocCudaArray(int nrows, int ncols);
F2D* fMallocCudaArray_copy(F2D* copy);
cudaError_t fCopyToGPU(F2D* host, F2D* device);
cudaError_t fCopyFromGPU(F2D* host, F2D* device);
F2D* fMallocAndCopy(F2D* host_array);
cudaError_t fCopyAndFree(F2D* device_array, F2D* host_array);
I2D* iMallocCudaArray(int nrows, int ncols);
I2D* iMallocCudaArray_copy(I2D* copy);
cudaError_t iCopyToGPU(I2D* host, I2D* device);
cudaError_t iCopyFromGPU(I2D* host, I2D* device);
I2D*  iMallocAndCopy(I2D* host_array);
cudaError_t iCopyAndFree(I2D* device_array, I2D* host_array);

//timing stuff
unsigned int* cudaStartTransfer();
void cudaEndTransfer(unsigned int* start);

#endif
