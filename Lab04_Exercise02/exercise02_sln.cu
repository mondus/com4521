#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Ex 2.2 (1/2), Update N to 2050
#define N 2050
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);
void vectorAddCPU(int *a, int *b, int *c);
int validate(int *a, int *ref);

// Ex 2.1 (2/2), 'a[i] - b[i]' has been corrected to 'a[i] + b[i]'
__global__ void vectorAdd(int *a, int *b, int *c, int max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// Ex 2.3, Extra threads can either skip the code, or return early
	// Forgetting to do this, would lead to out of bounds memory accesses
	if (i<max)
		c[i] = a[i] + b[i];
}



int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	// Launch add() kernel on GPU
	// Ex 2.2 (2/2) If the number of threads required does not divide by the number of blocks
	// it may be necessary to launch an extra block with unwanted threads
	dim3 blocksPerGrid((unsigned int)ceil(N / (double)THREADS_PER_BLOCK), 1, 1);
	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
	vectorAdd << <blocksPerGrid, threadsPerBlock >> >(d_a, d_b, d_c, N);
	checkCUDAError("CUDA kernel");

	//perform CPU version
	vectorAddCPU(a, b, c_ref);


	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	//validate
	errors = validate(c, c_ref);
	printf("CUDA GPU result has %d errors.\n", errors);



	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup");

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int *a)
{
	for (unsigned int i = 0; i < N; i++){
		a[i] = rand();
	}
}

void vectorAddCPU(int *a, int *b, int *c)
{
	for (int i = 0; i < N; i++){
		c[i] = a[i] + b[i];
	}
}
// Ex 2.1 (1/2), Simply iterate the 2 arrays and compare their values
int validate(int *a, int *ref){
	int errors = 0;
	for (int i = 0; i < N; i++){
		if (a[i] != ref[i]){
			errors++;
			fprintf(stderr, "ERROR at index %d: GPU result %d does not match CPU value of %d\n", i, a[i], ref[i]);
		}
	}
	return errors;
}