#include <stdio.h>
#include "testbench.h"

// Declare read-only constant array of ints
__constant__ int constantData[16384];

// Element-wise multiplication kernel
__global__ void multiply(int* arr, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (1) {
    if (idx < size) {
      arr[idx] = 2*constantData[idx];
    }
  }
}

int main() {
  // For SAFE macro
  cudaError_t err;

  // Initialize memory to be stored in constant memory
  int size = 16384;
  int hostData[16384];
  for (int i=0; i<16384; i++) {
    hostData[i] = 2*i;
  }

  // Print first 10 elements of input
  // for (int i=0; i<10; i++) {
  //   printf("%d ", hostData[i]);
  // }
  // printf("\n");

  // Copy data to constant memory
  SAFE(cudaMemcpyToSymbol(constantData, hostData, sizeof(int)*16384));

  // Copy host data to global memory as well
  int* d_x;
  SAFE(cudaMalloc(&d_x, size*sizeof(int)));
  SAFE(cudaMemcpy(d_x, hostData, sizeof(int)*size, cudaMemcpyHostToDevice));

  // Launch multiplication kernel
  multiply<<<16, 1024>>>(d_x, size);

  // Copy result back
  SAFE(cudaMemcpy(hostData, d_x, sizeof(int)*size, cudaMemcpyDeviceToHost));

  // Print first 10 elements of result
  // for (int i=0; i<10; i++) {
  //   printf("%d ", hostData[i]);
  // }
  // printf("\n");
  return 0;
}
