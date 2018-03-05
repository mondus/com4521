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

#define BLOCK_SIZE 8
#define NUM_SUBS (A_WIDTH / BLOCK_SIZE)

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

__global__ void matrixMulCUDA()
{
    // Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = bx*BLOCK_SIZE + tx;
	int y = by*BLOCK_SIZE + ty;
    

	float Csub = 0;
	//iterate A_WIDTH (same as B_HEIGHT) to calculate the product
	for (int k = 0; k < A_WIDTH; k++){
		Csub += d_A[y][k] * d_B[k][x]; 
	}

	// Store the product value of C matrix
	d_C[y][x] = Csub;
}

__global__ void matrixMulCUDASharedMemory()
{
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

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

		As[ty][tx] = d_A[a_y][a_x];		// global memory load with stride of 1, SM bank with stride of 1 :-)
		Bs[ty][tx] = d_B[b_y][b_x];		// global memory load with stride of 1, SM bank with stride of 1 :-)

		// Sync to ensure sub matrix is fully loaded
		__syncthreads();

		// sum products of A and B sub matrices
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[ty][k] * Bs[k][tx];		// conflict free loads regardless of BLOCK_SIZE due to bradcast read and stride of 1
		}

		// Sync to prevent run ahead (blocks loading new SM values before others have completed)
		__syncthreads();
	}

	// Store the product value of C matrix
	int c_x = (bx*BLOCK_SIZE) + tx;
	int c_y = (by*BLOCK_SIZE) + ty;
	d_C[c_y][c_x] = Csub;	// global memory load with stride of 1 :-)
}


int main(int argc, char **argv)

{
	unsigned int mem_size_A, mem_size_B, mem_size_C;
	unsigned int x, y, errors;
	float msec, occupancy;
	cudaEvent_t start, stop;

	if (A_WIDTH != B_HEIGHT){
		printf("Error: A_HEIGHT and B_WIDTH do not match\n");
	}

	mem_size_A = sizeof(float)* A_WIDTH* A_HEIGHT;
	mem_size_B = sizeof(float)* B_WIDTH* B_HEIGHT;
	mem_size_C = sizeof(float)* C_WIDTH* C_HEIGHT;

	// Initialise A
	for (y = 0; y < A_HEIGHT; y++)
	for (x = 0; x <A_WIDTH; x++)
		h_A[y][x] = (float)rand() / RAND_MAX;
	// Initialise B
	for (y = 0; y < B_HEIGHT; y++)
	for (x = 0; x <B_WIDTH; x++)
		h_B[y][x] = (float)rand() / RAND_MAX;

	// copy host memory to device
	cudaMemcpyToSymbol(d_A, h_A, mem_size_A);
	cudaMemcpyToSymbol(d_B, h_B, mem_size_B);
	checkCUDAError("CUDA memcpy");

	// Allocate CUDA events that we'll use for timing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	checkCUDAError("CUDA event creation");

	// Setup execution parameters
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(C_WIDTH / BLOCK_SIZE, C_HEIGHT / BLOCK_SIZE);
	cudaEventRecord(start);
	
    
    //matrixMulCUDA << < grid, threads >> >();
    //Comment out the above line and complete the shared memory version of the kernel
    matrixMulCUDASharedMemory << < grid, threads >> >();
    
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	checkCUDAError("CUDA kernel execution and timing");

	cudaEventElapsedTime(&msec, start, stop);
	cudaThreadSynchronize();
	checkCUDAError("CUDA timing");

	// Compute the ocupancy
	occupancy = 0;

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

	printf("Kernel time was %f with theoretical occupancy of %f\n", msec, occupancy);

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
