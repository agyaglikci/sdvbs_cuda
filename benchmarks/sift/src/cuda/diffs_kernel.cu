#include "sift.h"
#include "cuda_sift.h"


__global__ void diffs_kernel(F2D* in1, F2D* in2, F2D* out)
{
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = in2->width;
    int height = in2->height;

    if (tidX < width && tidY < height)
        out->data[tidY * width + tidX] = in1->data[tidY * width + tidX] - in2->data[tidY * width + tidX];
}