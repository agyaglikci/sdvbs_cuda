#include "sift.h"
#include "cuda_sift.h"

__global__ void halfSize_kernel(F2D * in, F2D * out, int width, int height)
{
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // if (row < height)
    // {
    //     for(int i = 0; i < width; i++)
    //     {
    //         out->data[row * width + i] = in->data[row * 4 * width + 2 * i];
    //     }
    // }
    // if (row == 0)
    // {
    //     out->width = width;
    //     out->height = height;
    // }
    // int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    // int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    // if (tidX < width && tidY < height)
    //     out->data[tidY * width + tidY] = in->data[tidY * 2 * width + tidY * 2];

    // if (tidX + tidY == 0)
    // {
    //     out->width = width;
    //     out->height = height;
    // }
    out->width = width;
    out->height = height;
    int k = 0;
    for(int i = 0; i < 2*height; i+=2)
    {
        for(int j = 0; j < 2*width; j+=2)
        {
            out->data[k++] = in->data[i * 2 * width + j];
        }
    }
}
