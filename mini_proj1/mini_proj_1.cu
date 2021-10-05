#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#define BLOCK_SIZE 32

// template <int BLOCK_SIZE> //for ease of use
__global__ void MatrixVectorMulCUDA(float *output, float *A, float *x, 
                                int nRows, int nCols)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // utilize shared memory on device for speedup
    __shared__ float shared_x[BLOCK_SIZE];

    float tmp = 0.0f;
    
    for (uint i=0; i<((nCols + BLOCK_SIZE - 1)/BLOCK_SIZE); ++i) {
        if ((i * BLOCK_SIZE + threadIdx.x) < nCols) //if thread in width
            //load element onto shared memory vector
            shared_x[threadIdx.x] = x[i*BLOCK_SIZE + threadIdx.x];
        else
            shared_x[threadIdx.x] = 0.0f; //else zero
        __syncthreads();

        #pragma unroll
        for (uint j=0; j < BLOCK_SIZE; ++j) {
            // A[i][j] * x[j] basically
            tmp += A[tid + (BLOCK_SIZE*i + j) * nRows] * shared_x[j];
        }
        __syncthreads();
    }

    if (tid<nRows)
        output[tid] = tmp;
}

void ConstantInt(float *data, int size, float val) {
    for (int i=0; i<size; ++i) 
        data[i] = val;
}

int main(void)
{
    int N = 20;
    int M = 25;
    int nRows = N;
    int nCols = M;

    //allocating host memory
    float *h_vec = (float*)malloc(sizeof(float)*N);   // vector dim (Nx1)
    if (h_vec == NULL)
    {
        fprintf(stderr, "Failed to allocate host vector.\n");
        exit(1);
    }
    float *h_mat = (float*)malloc(sizeof(float)*N*M); // matrix dim (NxM)
    if (h_mat == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix.\n");
        exit(1);
    }
    float *h_out = (float*)malloc(sizeof(float)*M);   // output dim (1xM)
    if (h_out == NULL)
    {
        fprintf(stderr, "Failed to allocate host output.\n");
        exit(1);
    }

    //initialize host memory
    const float valA = 1.0f;
    const float valB = 0.01f;
    ConstantInt(h_vec, sizeof(float)*N, valA);
    ConstantInt(h_mat, sizeof(float)*N*M, valB);
    
    // allocate device memory
    float *d_vec, *d_mat, *d_out;
    checkCudaErrors(cudaMalloc((void **)(&d_vec), sizeof(float)*N));
    checkCudaErrors(cudaMalloc((void **)(&d_mat), sizeof(float)*N*M));
    checkCudaErrors(cudaMalloc((void **)(&d_out), sizeof(float)*M));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_vec, h_vec, sizeof(float)*N, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_mat, h_mat, sizeof(float)*N*M, cudaMemcpyHostToDevice));

    //parameters
    dim3 dim_grid( (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dim_block(BLOCK_SIZE);

    MatrixVectorMulCUDA<<<dim_grid, dim_block>>>(d_out, d_mat, d_vec, N, M);

    cudaDeviceSynchronize();

    // copy results from device to host
    checkCudaErrors(cudaMemcpy(h_out, d_out, M*sizeof(float), cudaMemcpyDeviceToHost));

    printf("checking computed result for correctness: ");
    for (int i=0; i<M; i++) {
        if (1e-7 < fabs(h_out[i] - 0.01)) {
            printf("error with matrix accuracy/correctness.\n");
        }
    }

    //clean up memory
    free(h_vec);
    free(h_mat);
    free(h_out);
    checkCudaErrors(cudaFree(d_vec));
    checkCudaErrors(cudaFree(d_mat));
    checkCudaErrors(cudaFree(d_out));

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess)
        printf("CUDA Error: %s;", cudaGetErrorString(err));
    cudaProfilerStop();
    return 0;


}