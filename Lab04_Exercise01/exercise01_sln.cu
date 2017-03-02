#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//The number of character in the encrypted text
#define N 1024

void checkCUDAError(const char*);
void read_encrypted_file(int*);

#define A 15
#define B 27
#define M 128
#define A_MMI_M 111


__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void affine_decrypt(int *d_input, int *d_output)
{
	int index = threadIdx.x;
	int value = d_input[index];
	value = modulo(A_MMI_M * (value - B), M);
	d_output[index] = value;
}

__global__ void affine_decrypt_multiblock(int *d_input, int *d_output)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int value = d_input[index];
	value = modulo(A_MMI_M * (value - B), M);
	d_output[index] = value;
}


int main(int argc, char *argv[])
{
	int *h_input, *h_output;
	int *d_input, *d_output;
	unsigned int size;
	int i;

	size = N * sizeof(int);

	/* allocate the host memory */
	h_input = (int *)malloc(size);
	h_output = (int *)malloc(size);

	/* allocate device memory */
	cudaMalloc((void **)&d_input, size);
	cudaMalloc((void **)&d_output, size);
	checkCUDAError("Memory allocation");

	/* read the encryted text */
	read_encrypted_file(h_input);

	/* copy host input to device input */
	cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
	checkCUDAError("Input transfer to device");

	/* Configure the grid of thread blocks and run the GPU kernel */
	dim3 blocksPerGrid(8, 1, 1);
	dim3 threadsPerBlock(N / 8, 1, 1);
	affine_decrypt_multiblock << <blocksPerGrid, threadsPerBlock >> >(d_input, d_output);

	/* wait for all threads to complete */
	cudaThreadSynchronize();
	checkCUDAError("Kernel execution");

	/* copy the gpu output back to the host */
	cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
	checkCUDAError("Result transfer to host");

	/* print out the result to screen */
	for (i = 0; i < N; i++) {
		printf("%c", (char)h_output[i]);
	}
	printf("\n");

	/* free device memory */
	cudaFree(d_input);
	cudaFree(d_output);
	checkCUDAError("Free memory");

	/* free host buffers */
	free(h_input);
	free(h_output);

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

void read_encrypted_file(int* input)
{
	FILE *f = NULL;
	f = fopen("encrypted01.bin", "rb"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find encrypted01.bin file \n");
		exit(1);
	}
	//read encrypted data
	fread(input, sizeof(unsigned int), N, f);
	fclose(f);
}