#include <cuda.h>
#include <stdio.h>
#include "testbench.h"
#define PAGES 100000
#define PAGE_SIZE 65536

__global__ void dereference(int *a) {
	a[threadIdx.x] = 1;
}

int
main() {
	// init driver api
	SAFE_D(cuInit(0));

	// create a context on the gpu
	CUdevice device;
	SAFE_D(cuDeviceGet(&device, 0));
	CUcontext context;
	SAFE_D(cuCtxCreate(&context, 0, device));

	// allocate device memory
	CUdeviceptr d_a[PAGES];
	for (int i=0; i<PAGES; i++) {
		SAFE_D(cuMemAlloc(&d_a[i], PAGE_SIZE));
	}

	for (int i=0; i<PAGES; i++) {
		// dereference memory address
		dereference<<<1, 1024>>>((int*) NULL);
	}

	// sync to catch an error
	PRINT_ERROR(cudaDeviceSynchronize());
	int *d_test;
	PRINT_ERROR(cudaMalloc(&d_test, 10*4));
	for (int i=0; i<PAGES; i++) {
		cuMemFree(d_a[i]);
	}
	return 0;
}
