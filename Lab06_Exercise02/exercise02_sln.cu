#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define A_WIDTH 1024
#define A_HEIGHT 1024
#define B_WIDTH 1024
#define B_HEIGHT 1024
#define C_WIDTH B_WIDTH
#define C_HEIGHT A_HEIGHT

#define NUM_SUBS (A_WIDTH / BLOCK_SIZE)

#define EPSILON 0.001f

__device__ float d_A[A_HEIGHT][A_WIDTH];
__device__ float d_B[B_HEIGHT][B_WIDTH];
__device__ float d_C[C_HEIGHT][C_WIDTH];

float h_A[A_HEIGHT][A_WIDTH];
float h_B[B_HEIGHT][B_WIDTH];
float h_C[C_HEIGHT][C_WIDTH];
float h_C_ref[C_HEIGHT][C_WIDTH];

void checkCUDAError(const char *msg);
void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[B_HEIGHT][B_WIDTH], float C[C_HEIGHT][C_WIDTH]);
int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]);
int requiredSM(int tpb);

__constant__ int BLOCK_SIZE;

__global__ void matrixMulCUDA()
{
	extern __shared__ float sm[];
	float *As = &sm[0];
	float *Bs = &sm[BLOCK_SIZE*BLOCK_SIZE];

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float Csub = 0;

	for (int i = 0; i < NUM_SUBS; i++){
		// Calculate indices of A and B matrix required to load the shared block of memory
		int a_x = (i*BLOCK_SIZE) + tx;
		int a_y = (by*BLOCK_SIZE) + ty;
		int b_x = (bx*BLOCK_SIZE) + tx;
		int b_y = (i*BLOCK_SIZE) + ty;

		As[(ty*BLOCK_SIZE) + tx] = d_A[a_y][a_x];
		Bs[(ty*BLOCK_SIZE) + tx] = d_B[b_y][b_x];

		// Sync to ensure sub matrix is fully loaded
		__syncthreads();

		// sum products of A and B sub matrices
		for (int k = 0; k < BLOCK_SIZE; k++)
		{
			Csub += As[(ty*BLOCK_SIZE) + k] * Bs[(k*BLOCK_SIZE) + tx];
		}

		// Sync to prevent run ahead (blocks loading new SM values before others have completed)
		__syncthreads();
	}

	// Store the product value of C matrix
	int c_x = (bx*BLOCK_SIZE) + tx;
	int c_y = (by*BLOCK_SIZE) + ty;
	d_C[c_y][c_x] = Csub;
}


int main(int argc, char **argv)
{
	unsigned int mem_size_A, mem_size_B, mem_size_C;
	unsigned int x, y, errors;
	int maxActiveBlocks, TPB, min_grid_size, block_size;
	float msec, occupancy;
	cudaDeviceProp props;
	cudaEvent_t start, stop;

	if (A_WIDTH != B_HEIGHT){
		printf("Error: A_HEIGHT and B_WIDTH do not match\n");
	}

	mem_size_A = sizeof(float)* A_WIDTH* A_HEIGHT;
	mem_size_B = sizeof(float)* B_WIDTH* B_HEIGHT;
	mem_size_C = sizeof(float)* C_WIDTH* C_HEIGHT;

	// Initialise A
	for (x = 0; x <A_WIDTH; x++)
	for (y = 0; y < A_HEIGHT; y++)
		h_A[x][y] = (float)rand() / RAND_MAX;
	// Initialise B
	for (x = 0; x <B_WIDTH; x++)
	for (y = 0; y < B_HEIGHT; y++)
		h_B[x][y] = (float)rand() / RAND_MAX;

	// copy host memory to device
	cudaMemcpyToSymbol(d_A, h_A, mem_size_A);
	cudaMemcpyToSymbol(d_B, h_B, mem_size_B);
	checkCUDAError("CUDA memcpy");

	// Allocate CUDA events that we'll use for timing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	checkCUDAError("CUDA event creation");

	// Calculate the block size
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&min_grid_size, &TPB, matrixMulCUDA, requiredSM, 0);
	TPB = (int)pow(4, floor(log(TPB) / log(4))); //round to nearest square power 2
	block_size = (int) sqrt(TPB);
	cudaMemcpyToSymbol(BLOCK_SIZE, &block_size, sizeof(int));

	// calculate grid size and execute kernel
	dim3 threads(block_size, block_size);
	dim3 grid(C_WIDTH / block_size, C_HEIGHT / block_size);
	cudaEventRecord(start);
	matrixMulCUDA << < grid, threads, requiredSM(block_size*block_size) >> >();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	checkCUDAError("CUDA kernel execution and timing");

	cudaEventElapsedTime(&msec, start, stop);
	cudaThreadSynchronize();
	checkCUDAError("CUDA timing");

	// Compute the ocupancy
	cudaGetDeviceProperties(&props, 0);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, matrixMulCUDA, block_size*block_size, 0);
	occupancy = (maxActiveBlocks * block_size*block_size) / (float)(props.maxThreadsPerMultiProcessor);

	// Copy result from device to host
	cudaMemcpyFromSymbol(h_C, d_C, mem_size_C);
	checkCUDAError("CUDA memcpy results");

	// Compute reference CPU version
	matrixMulCPU(h_A, h_B, h_C_ref);

	// Check for errors
	errors = matrixMulTest(h_C, h_C_ref);
	if (errors)
		printf("%d total errors\n", errors);
	else
		printf("Test passed successfully\n");

	printf("Kernel time was %f with block size %d and theoretical occupancy of %f\n", msec, block_size, occupancy);

}

void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[C_HEIGHT][C_WIDTH], float C[C_HEIGHT][C_WIDTH])
{
	int x, y, k;
	for (y = 0; y < C_HEIGHT; y++){
		for (x = 0; x < C_WIDTH; x++){
			C[y][x] = 0;
			for (k = 0; k < A_WIDTH; k++){
				C[y][x] += A[y][k] * B[k][x];
			}
		}
	}

}

int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH])
{
	int errors = 0;
	int y, x;

	for (y = 0; y < C_HEIGHT; y++){
		for (x = 0; x < C_WIDTH; x++){
			if (C[y][x] != Cref[y][x]){
				errors++;
				printf("Device item c[%d][%d] = %f does not mach host result %f\n", y, x, C[y][x], Cref[y][x]);
			}
		}
	}

	return errors;
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

int requiredSM(int tpb){
	return (tpb*sizeof(float)* 2);
}
