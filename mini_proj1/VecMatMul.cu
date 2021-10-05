#include <cuda_profiler_api.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define BLOCK_SIZE 256
//#define Ni 4096  //input layers
//#define Nn 1024  //output layers

__global__
void kernel(float *vec, float *mat, float *out, const int Ni, const int Nn){
    const unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
    __shared__ float shared_vec[12000];
    float sum=0.0f;
    if(tid<Nn){
        shared_vec[tid] = vec[tid];
        __syncthreads();
#pragma unroll
        for(int i=0; i<Ni; i++)
            sum += shared_vec[i]*mat[(i*Nn)+tid];
        __syncthreads();
        out[tid]=sum;
    }
}

// helper functions
void init_array(float *a, const int Ni, const int val);
void init_mat(float *a, const int Ni, const int Nn, const int val);
void print_array(float *a, const int Ni, char *d);
void print_mat(float *a, const int Ni, const int Nn, char *d);

int main (void) {
    srand( time(NULL) );

    //int Ni = 25088;
    //int Nn = 4096;
    int Ni = 4096;
    int Nn = 1024;

    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    a=(float*)malloc(sizeof(float)*Ni);
    b=(float*)malloc(sizeof(float)*Ni*Nn);
    c=(float*)malloc(sizeof(float)*Nn);
    init_array(a, Ni, 1.0f);
    init_mat(b, Ni, Nn, 2.0f);
    init_array(c, Nn, 0.0f);

/*    printf("<<<<<<<<<< initial data:\n");
    print_array(a, Ni, "in-vector");
    print_mat(b, Ni, Nn, "matrix");
    print_array(c, Nn, "out-vector");
*/

    //allocate device memory
    cudaMalloc((void**)&dev_a, sizeof(float)*Ni);
    cudaMalloc((void**)&dev_b, sizeof(float)*Ni*Nn);
    cudaMalloc((void**)&dev_c, sizeof(float)*Nn);

    //transfer host to device memory
    cudaMemcpy(dev_a, a, sizeof(float)*Ni, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(float)*Ni*Nn, cudaMemcpyHostToDevice);

//    printf("\n\nRunning Kernel...\n\n");
    kernel<<<Nn/256+1, 256>>>(dev_a, dev_b, dev_c, Ni, Nn);
    //printf("error code: %s\n",cudaGetErrorString(cudaGetLastError()));
    //cudaDeviceSynchronization();

    //get output from device to host
    cudaMemcpy(c, dev_c, sizeof(float)*Nn, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);

//    printf(">>>>>>>>>> final data:\n");
//    print_array(c, Nn, "out-vector");

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess)
        printf("CUDA Error: %s;", cudaGetErrorString(err));
    cudaProfilerStop();
    return 0;
};

void init_array(float *a, const int Ni, const int val) {
        int i;
        for(i=0; i<Ni; i++)
                a[i] = val;
}
void init_mat(float *a, const int Ni, const int Nn, const int val) {
        int i, j;
        for(i=0; i<Ni; i++)
            for(j=0; j<Nn; j++)
                    a[i*Nn+j] = val;
}
void print_array(float *a, const int Ni, char *d) {
        int i;
        for(i=0; i<Ni; i++)
                printf("\n%s[%d]: %f",d, i, a[i]);
    printf("\n");
}
void print_mat(float *a, const int Ni, const int Nn, char *d) {
        int i, j;
        for(i=0; i<Ni; i++){
        printf("\n%s[%d]:", d, i);
        for (j=0; j<Nn; j++)
                    printf("\t%6.4f", a[i*Nn+j]);
    }
    printf("\n");
}
