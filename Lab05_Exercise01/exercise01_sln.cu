
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 4194304
#define THREADS_PER_BLOCK 128
#define PUMP_RATE 2

#define READ_BYTES N*(2*4)  //2 reads of 4 bytes (a and b)
#define WRITE_BYTES N*(4*1) //1 write of 4 bytes (to c)

__device__ float d_a[N];
__device__ float d_b[N];
__device__ float d_c[N];

void random_floats(float *a);

__global__ void vectorAdd() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	d_c[i] = d_a[i] + d_b[i];
}

int main(void) {
	float *a, *b, *c;		// host copies of a, b, c
	int size = N * sizeof(float);
	cudaEvent_t start, stop;
	float milliseconds = 0;
	int deviceCount = 0;
	double theoretical_BW;
	double measure_BW;


	cudaGetDeviceCount(&deviceCount);
	if (deviceCount > 0)
	{
		cudaSetDevice(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		theoretical_BW = deviceProp.memoryClockRate * PUMP_RATE * (deviceProp.memoryBusWidth / 8.0) / 1e6; //convert to GB/s
	}


	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	a = (float *)malloc(size); random_floats(a);
	b = (float *)malloc(size); random_floats(b);
	c = (float *)malloc(size);

	cudaMemcpyToSymbol(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_b, b, size, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	vectorAdd << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	measure_BW = (READ_BYTES + WRITE_BYTES) / (milliseconds * 1e6);

	cudaMemcpyFromSymbol(c, d_c, size, cudaMemcpyDeviceToHost);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	printf("Execution time is %f ms\n", milliseconds);
	printf("Theoretical Bandwidth is %f GB/s\n", theoretical_BW);
	printf("Measured Bandwidth is %f GB/s\n", measure_BW);
	return 0;
}

void random_floats(float *a)
{
	for (unsigned int i = 0; i < N; i++){
		a[i] = (float)rand()/RAND_MAX;
	}
}
