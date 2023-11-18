#include <stdio.h>
#include "testbench.h"

__global__ void spin() {
    while (1) {}
}

int main()  {
    cudaError_t err;
    spin<<<40, 1024>>>();
    SAFE(cudaDeviceSynchronize());
    printf("Exited\n");
    return 0;
}
