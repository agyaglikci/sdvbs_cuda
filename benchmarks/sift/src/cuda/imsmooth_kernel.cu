#include "sift.h"
#include "cuda_sift.h"

__global__ void imsmooth_kernel(F2D* input, F2D* output, int filterSize, int inTileSize, int outTileSize, float * filter)
{
    extern __shared__ float input_s[];
    __shared__ float filter_s[1000];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * outTileSize + ty;
    int col_o = blockIdx.x * outTileSize + tx;

    int row_i = row_o - filterSize / 2;
    int col_i = col_o - filterSize / 2;

    int width = input->width;
    int height = input->height;

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
        input_s[ty * inTileSize + tx] = input->data[row_i * width + col_i];
    else
        input_s[ty * inTileSize + tx] = 0.0f;

    __syncthreads();

    float sum = 0.0f;
    if ((ty < outTileSize) && (tx < outTileSize))
    {
        for (int i = 0; i < filterSize; i++)
        {
            for (int j = 0; j < filterSize; j++)
            {
                sum += filter[i * filterSize + j] * input_s[(i + ty) * inTileSize + (j + tx)];
            }
        }

        if ((row_o < height) && (col_o < width))
            output->data[row_o * width + col_o] = sum;
    }
}
