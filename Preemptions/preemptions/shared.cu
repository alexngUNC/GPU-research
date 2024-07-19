#include <stdio.h>
#include "testbench.h"
#define CONCURRENT_TB 38
#define PERCENTAGE_SHARED (2.0/2.0)
#define SHARED_MEM_TB 49152


__global__ void vecAdd(float *a, float *b, int n, int *flag, int *blockCounter) {
	// global thread index
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	// dynamic shared memory for result
	extern __shared__ float res[];

	// index for shared memory portion
	int idx = (int) 12.0 * PERCENTAGE_SHARED * ((double) threadIdx.x);

	// loop to fill up shared memory
	b[i] += a[i];
	int limit = (int) 12.0 * PERCENTAGE_SHARED;
	for (int j=0; j<limit; j++) {
		res[idx + j] = b[i];
	}

	// ensure all blocks have loaded their portion of shared memory
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(blockCounter, 1);
	}
	__syncthreads();

	// tell CPU that shared memory is fully saturated
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		while (*blockCounter < gridDim.x) {
			// wait for all blocks to load shared memory
		}
		*flag = 0;
	}

	// spin with desired shared memory usage
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

	// flag for synchronization
	int *flag;
	SAFE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped));
	*flag = 1;

	// block counter for syncrhonizing across blocks
	int *blockCounter;
	SAFE(cudaMalloc(&blockCounter, sizeof(int)));
	SAFE(cudaMemset(blockCounter, 0, sizeof(int)));

	// launch kernel
	int sharedMem = SHARED_MEM_TB * PERCENTAGE_SHARED;
	vecAdd<<<CONCURRENT_TB, 1024, sharedMem>>>(d_a, d_b, n, flag, blockCounter);
	while (*flag) { /* wait until all of shared memory is loaded */ }
	printf("Shared memory is fully saturated!\n");
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
