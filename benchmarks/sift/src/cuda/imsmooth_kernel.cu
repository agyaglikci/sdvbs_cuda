#include "sift.h"
#include "cuda_sift.h"

__global__ void imsmooth_kernel(F2D* input, F2D* output, int filterSize, int inTileSize, int outTileSize, float * filter)
{
    extern __shared__ float input_s[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * outTileSize + ty;
    int col_o = blockIdx.x * outTileSize + tx;

    int row_i = row_o - filterSize / 2;
    int col_i = col_o - filterSize / 2;

    int inWidth = input->width;
    int inHeight = input->height;

    int outWidth = inWidth;
    int outHeight = inHeight;

    // if (halfSize > 0)
    // {
    //      outWidth = (outWidth + 1) / 2;
    //      outHeight = (outHeight + 1) / 2;
    //      row_i *= 2;
    //      col_i *= 2;
    // }

    if ((row_i >= 0) && (row_i < inHeight) && (col_i >= 0) && (col_i < inWidth))
        input_s[ty * inTileSize + tx] = input->data[row_i * inWidth + col_i];
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

        if ((row_o < outHeight) && (col_o < outWidth))
            output->data[row_o * outWidth + col_o] = sum;

        if (tx + ty == 0)
        {
            output->width = outWidth;
            output->height = outHeight;
        }
    }
}

__global__ void imsmoothRow_kernel (F2D* d_Result, F2D* d_Data, int radius, int tileW, float * filter, int dataW, int dataH)
{
    extern __shared__ float data[];
    const int xPos = threadIdx.x + blockIdx.x * blockDim.x;
    const int yPos = threadIdx.y + blockIdx.y * blockDim.y;
    const int gLoc = xPos + yPos * dataW;

    int x;

    const int x0 = threadIdx.x + (blockIdx.x * blockDim.x);
    const int shift = threadIdx.y * (tileW + radius * 2);

    x = x0 - radius;
    if (x < 0 || x >= dataW)
        data[threadIdx.x + shift] = 0;
    else
        data[threadIdx.x + shift] = d_Data->data[gLoc - radius];

    x = x0 + radius;
    if (x > dataW - 1)
        data[threadIdx.x + 2 * radius + shift] = 0;
    else
        data[threadIdx.x + 2 * radius + shift] = d_Data->data[gLoc + radius];

    __syncthreads();
    if (xPos < dataW && yPos < dataH)
    {
        float sum = 0.0f;
        x = radius + threadIdx.x;
        for (int i = -radius; i <= radius; i++)
            sum += data[x + i + shift] * filter[radius + i];

        d_Result->data[gLoc] = sum;
        if (gLoc == 0)
        {
            d_Result->width = dataW;
            d_Result->height = dataH;
        }
    }
}

__global__ void imsmoothCol_kernel (F2D* d_Result, F2D* d_Data, int radius, int tileW, float * filter, int dataW, int dataH)
{
    extern __shared__ float data[];

    const int xPos = threadIdx.x + blockIdx.x * blockDim.x;
    const int yPos = threadIdx.y + blockIdx.y * blockDim.y;
    const int gLoc = xPos + yPos * dataW;

    int y;

    const int y0 = threadIdx.y + (blockIdx.y * blockDim.y);
    const int shift = threadIdx.y * tileW;

    y = y0 - radius;
    if (y < 0 || y >= dataH)
        data[threadIdx.x + shift] = 0;
    else
        data[threadIdx.x + shift] = d_Data->data[gLoc - (dataW * radius)];

    y = y0 + radius;
    const int shiftl = shift + (2 * radius * tileW);
    if (y > dataH-1)
        data[threadIdx.x + shiftl] = 0;
    else
        data[threadIdx.x + shiftl] = d_Data->data[gLoc + (dataW * radius)];

    __syncthreads();
    if (xPos < dataW && yPos < dataH)
    {
        float sum = 0.0f;
        for (int i = 0; i <= radius*2; i++)
            sum += data[threadIdx.x + (threadIdx.y + i) * tileW] * filter[i];

        d_Result->data[gLoc] = sum;
        if (gLoc == 0)
        {
            d_Result->width = dataW;
            d_Result->height = dataH;
        }
    }

}