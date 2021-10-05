#include <iostream>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>


#define MASK_WIDTH 3  //Kx and Ky
#define MASK_RADIUS MASK_WIDTH / 2
#define TILE_WIDTH 8
#define W (TILE_WIDTH + MASK_WIDTH - 1)

__global__ void Convolution3D(float *input, float* mask, float *result, int width, int height, int depth)
{
    //allocating shared memory for intermediate ops
    __shared__ float shared_mem[W][W][W];

    // First batch loading
    int out = (threadIdx.y * TILE_WIDTH) + threadIdx.x
                     + (threadIdx.z * TILE_WIDTH * TILE_WIDTH);
    int out_tmp = out;
    int outX = out_tmp % W;
    out_tmp = out_tmp / W;
    int outY = out_tmp % W;
    out_tmp = out_tmp / W;
    int outZ = out_tmp;

    // input's indices
    int srcZ = outZ + (blockIdx.z * TILE_WIDTH) - MASK_RADIUS;
    int srcY = outY + (blockIdx.y * TILE_WIDTH) - MASK_RADIUS;
    int srcX = outX + (blockIdx.x * TILE_WIDTH) - MASK_RADIUS;
    int src = srcX + (srcY * width) + (srcZ * width * height);

    // initializing shared memory
    if(srcZ >= 0 && srcZ < depth && srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        shared_mem[outZ][outY][outX] = input[src];
    else
        shared_mem[outZ][outY][outX] = 0;

    // Second batch loading
    out = threadIdx.x + (threadIdx.y * TILE_WIDTH) + (threadIdx.z * TILE_WIDTH * TILE_WIDTH) + TILE_WIDTH * TILE_WIDTH * TILE_WIDTH;
    out_tmp = out;
    outX = out_tmp % W;
    out_tmp = out_tmp / W;
    outY = out_tmp % W;
    out_tmp = out_tmp / W;
    outZ = out_tmp;

    srcZ = outZ + (blockIdx.z * TILE_WIDTH) - MASK_RADIUS;
    srcY = outY + (blockIdx.y * TILE_WIDTH) - MASK_RADIUS;
    srcX = outX + (blockIdx.x * TILE_WIDTH) - MASK_RADIUS;
    src = srcX + (srcY * width) + (srcZ * width * height);

    // if within tiling area
    if(outZ < W)
    {
        //transfer data to shared memory
        if(srcZ >= 0 && srcZ < depth && srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            shared_mem[outZ][outY][outX] = input[src];
        else
            shared_mem[outZ][outY][outX] = 0;
    }
    __syncthreads();

    //3d convolution
    int z;
    int y;
    int x;
    float sum = 0;
    for(z = 0; z < MASK_WIDTH; z++)
        for(y = 0; y < MASK_WIDTH; y++)
            for(x = 0; x < MASK_WIDTH; x++)
                sum += shared_mem[threadIdx.z + z][threadIdx.y + y][threadIdx.x + x] * mask[x + (y * MASK_WIDTH) + (z * MASK_WIDTH * MASK_WIDTH)];
    
    //check indices before loading into output
    z = threadIdx.z + (blockIdx.z * TILE_WIDTH);
    y = threadIdx.y + (blockIdx.y * TILE_WIDTH);
    x = threadIdx.x + (blockIdx.x * TILE_WIDTH);
    if(z < depth && y < height && x < width)
        result[x + (y * width) + (z * width * height)] = sum;

    __syncthreads();

}

void init_mat(float *a, const int Nx, const int Ny, const int Ni, const int val) {
    int i, j, k;
    for(i=0; i<Nx; i++)
        for(j=0; j<Ny; j++)
            for(k=0; k<Ni; k++)
                a[(i*Ny*Ni)+j*Ni+k] = val;
}

int main(int argc, char* argv[])
{
    int image_width  = 14;  //Nx
    int image_height = 14;  //Ny
    int image_depth  = 512;   //Ni

    /*
    int image_width  = 224;  //Nx
    int image_height = 224;  //Ny
    int image_depth  = 64;   //Ni
    */
    
    // initialize host mem
    float *d_in;
    float *d_out;
    float *d_mask;

    float *data = (float*)malloc(sizeof(float)*image_width*image_height*image_depth);
    init_mat(data, image_width, image_height, image_depth, 2.0f);

    float *mask = (float*)malloc(sizeof(float)*MASK_WIDTH*MASK_WIDTH*MASK_WIDTH);
    init_mat(mask, MASK_WIDTH, MASK_WIDTH, MASK_WIDTH, 1.0f);

    //allocate device mem
    int input_size = image_height * image_width * image_depth;
    int output_size = image_height * image_width * image_depth;
    cudaMalloc((void **)&d_in,  input_size * sizeof(float));
    cudaMalloc((void **)&d_out, output_size* sizeof(float));
    cudaMalloc((void **)&d_mask, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float));

    cudaMemcpy(d_in, data, image_width * image_height * image_depth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dim_grid((image_width + TILE_WIDTH - 1) / TILE_WIDTH, (image_height + TILE_WIDTH - 1) / TILE_WIDTH, (image_depth + TILE_WIDTH - 1) / TILE_WIDTH);
    Convolution3D<<<dim_grid, dim_block>>>(d_in, d_mask, d_out, image_width, image_height, image_depth);
    cudaDeviceSynchronize();

    //get data from device to host
    cudaMemcpy(data, d_out, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_mask);
    free(data);
    free(mask);

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess)
        printf("CUDA Error: %s;", cudaGetErrorString(err));
    cudaProfilerStop();
    return 0;
}