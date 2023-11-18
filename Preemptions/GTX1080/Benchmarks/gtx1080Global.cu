#include <stdio.h>
#include "testbench.h"
#define CONCURRENT_BLOCKS 40
#define FILL_SHARED_ITERS 12.0
#define PERCENTAGE_SHARED 1


// Kernel for computing aX + Y, then copying the result and adding 
__global__ void vecAdd(float a, float* x, float* y, float* z1, float* z2) {
    // Thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop index
    int index = (int) FILL_SHARED_ITERS * PERCENTAGE_SHARED * ((double) threadIdx.x);

    // Infinite loop so kernel uses full timeslice
    while (1) {
        // Y = aX + Y
        y[i] = a * x[i] + y[i];

        // Copy result to larger arrays to fill up desired memory
        int limit = (int) FILL_SHARED_ITERS * PERCENTAGE_SHARED;
        for (int j=0; j<limit; j++) {
            z1[index+j] = y[i];
            z2[index+j] = y[i];
            z2[index+j] += z1[index+j];
        }
        __syncthreads();

    }

}

int main() {
    // For SAFE error checking macro
    cudaError_t err;

    // Host pointers
    float *h_x, *h_y, *h_z1, *h_z2;

    // Number of concurrent threads
    int n = 40960; 

    // Allocate host memory
    h_x = (float*) malloc(n * sizeof(float));
    h_y = (float*) malloc(n * sizeof(float));
    h_z1 = (float*) malloc(n * sizeof(float) * FILL_SHARED_ITERS * PERCENTAGE_SHARED);
    h_z2 = (float*) malloc(n * sizeof(float) * FILL_SHARED_ITERS * PERCENTAGE_SHARED);
    
    // Check memory allocation
    if (h_x == NULL ||
        h_y == NULL ||
        h_z1 == NULL ||
        h_z2 == NULL) {
        fprintf(stderr, "Couldn't allocate host memory\n");
        return 1;
    }

    // Initialize vectors
    for (int i=0; i<=n; i++) {
        h_x[i-1] = i;
        h_y[i-1] = i+1;
    }

    // Device pointers
    float *d_x, *d_y, *d_z1, *d_z2;

    // Allocate device memory
    SAFE(cudaMalloc(&d_x, n * sizeof(float)));
    SAFE(cudaMalloc(&d_y, n * sizeof(float)));
    SAFE(cudaMalloc(&d_z1, n * sizeof(float) * FILL_SHARED_ITERS * PERCENTAGE_SHARED));
    SAFE(cudaMalloc(&d_z2, n * sizeof(float) * FILL_SHARED_ITERS * PERCENTAGE_SHARED));

    // Define a
    int a = 10.0;

    // Execute the kernel: 20 SMs, 2 TBs per SM
    vecAdd<<<40, 1024>>>(a, d_x, d_y, d_z1, d_z2);

    // Copy memory from device to host
    SAFE(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    SAFE(cudaFree(d_x));
    SAFE(cudaFree(d_y));

    // Free host memory
    free(h_x);
    free(h_y);
    free(h_z1);
    free(h_z2);
    return 0;
}
