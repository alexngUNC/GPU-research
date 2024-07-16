#include <cuda.h>
#include <stdio.h>
#include "testbench.h"

__global__ void dereference(int *a) {
	a[0] = 1;
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

	// dereference memory address
	dereference<<<1, 1>>>((int*) NULL);

	// sync for error
	SAFE(cudaDeviceSynchronize());
	return 0;
}
