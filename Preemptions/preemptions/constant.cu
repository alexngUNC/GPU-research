#include <stdio.h>
#include "testbench.h"
#define CONSTANT_MEMORY 65536
#define LENGTH 16384

__constant__ float constantData[LENGTH];

__global__ void vecAdd(float *a, int *flag) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	a[i] += constantData[i];
	*flag = 0;
	while (1) {}
}

int
main()
{
	// allocate host data for constant cache
	float hostData[LENGTH];
	for (int i=0; i<LENGTH; i++)
		hostData[i] = 2;

	// allocate host data for vector to be incremented
	float *h_a = (float *) malloc(CONSTANT_MEMORY);	
	for (int i=0; i<LENGTH; i++) {
		h_a[i] = 1;
	}
	
	// copy data to constant memory cache
	SAFE(cudaMemcpyToSymbol(constantData, hostData, CONSTANT_MEMORY));

	// copy data to device
	float *d_a;
	SAFE(cudaMalloc(&d_a, CONSTANT_MEMORY));
	SAFE(cudaMemcpy(d_a, h_a, CONSTANT_MEMORY, cudaMemcpyHostToDevice));

	// flag for synchronization
	int *flag;
	SAFE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped));
	*flag = 1;

	// spin on GPU once constant memory is accessed
	vecAdd<<<16, 1024>>>(d_a, flag);
	while (*flag) {}
	printf("Constant memory has been read!\n");
	SAFE(cudaDeviceSynchronize());

	// get result
	SAFE(cudaMemcpy(h_a, d_a, CONSTANT_MEMORY, cudaMemcpyDeviceToHost));
	for (int i=0; i<10; i++)
		printf("%f\t", h_a[i]);
	printf("\n");

	// free memory
	free(h_a);
	cudaFree(d_a);
	
	return 0;
}
