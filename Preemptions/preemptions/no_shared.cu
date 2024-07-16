#include <stdio.h>
#include "testbench.h"
#define CONCURRENT_TB 38
#define PERCENTAGE_SHARED (2.0/2.0)
#define SHARED_MEM_TB 49152


__global__ void vecAdd(float *a, float *b, int n, int *flag) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	b[i] += a[i];
	*flag = 0;
	while (1) {}
}


int
main()
{
	// vector length
	int n = CONCURRENT_TB * 1024;
	// host memory
	float *h_a, *h_b;
	h_a = (float *) malloc(n * sizeof(float));
	h_b = (float *) malloc(n * sizeof(float));
	for (int i=0; i<n; i++) {
		h_a[i] = 1;
		h_b[i] = 2;
	}

	// device memory
	float *d_a, *d_b;
	int bytes = n * sizeof(float);
	SAFE(cudaMalloc(&d_a, bytes));
	SAFE(cudaMalloc(&d_b, bytes));
	SAFE(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
	SAFE(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
	
	// flag for CPU synchronization
	int *flag;
	SAFE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped));
	*flag = 1;

	// launch kernel
	vecAdd<<<CONCURRENT_TB, 1024>>>(d_a, d_b, n, flag);
	while (*flag) {}
	printf("GPU is spinning!\n");
	SAFE(cudaDeviceSynchronize());

	// print result
	SAFE(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));
	for (int i=0; i<10; i++) {
		printf("%f\t", h_b[i]);
	}
	printf("\n");

	// free memory
	SAFE(cudaFree(d_a));
	SAFE(cudaFree(d_b));
	free(h_a);
	free(h_b);
	return 0;
}
