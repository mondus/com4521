// System includes
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust\device_ptr.h>
#include <thrust\scan.h>


#define TPB 256

#define NUM_PARTICLES 16384
#define ENV_DIM 32.0f
#define INTERACTION_RANGE 4.0f
#define ENV_BIN_DIM ((unsigned int)(ENV_DIM/INTERACTION_RANGE))
#define ENV_BINS (ENV_BIN_DIM*ENV_BIN_DIM)


struct key_values{
	int sorting_key[NUM_PARTICLES];
	int value[NUM_PARTICLES];
};
typedef struct key_values key_values;

struct particles{
	float2 location[NUM_PARTICLES];
	int nn_key[NUM_PARTICLES];
};
typedef struct particles particles;

struct environment{
	int count[ENV_BINS];
	int start_index[ENV_BINS];
};
typedef struct environment environment;

__device__ __host__ int2 binLocation(float2 location);
__device__ __host__ int binIndex(int2 bin);

void particlesCPU();
void particlesGPU();
void initParticles(particles *p);
int checkResults(char* name, particles *p);
void keyValuesCPU(particles *p, key_values *kv);
void sortKeyValuesCPU(key_values *kv);
void reorderParticlesCPU(key_values *kv, particles *p, particles *p_sorted);
void histogramParticlesCPU(particles *p, environment *env);
void prefixSumEnvironmentCPU(environment * env);

void checkCUDAError(const char *msg);


/* GPU Kernels */

__global__ void particleNNSearch(particles *p, environment *env)
{
	int2 bin;
	int i, x, y;
	int idx;
	float2 location;
	int nn;
	
	idx = blockIdx.x * blockDim.x + threadIdx.x;

	//get location
	location = p->location[idx];
	bin = binLocation(location);
	nn = -1;

	//check all neighbouring bins of particle (9 in total) - no boundary wrapping
	float dist_sq = ENV_DIM*ENV_DIM;	//a big number

	for (x = bin.x - 1; x <= bin.x + 1; x++){
		//no wrapping
		if ((x < 0) || (x >= ENV_BIN_DIM))			
			continue;

		for (y = bin.y - 1; y <= bin.y + 1; y++){
			//no wrapping
			if ((y < 0) || (y >= ENV_BIN_DIM))			
				continue;

			//get the bin index
			int bin_index = binIndex(make_int2(x, y));

			//get start index of the bin
			int bin_start_index = env->start_index[bin_index];

			//get the count of the bin
			int bin_count = env->count[bin_index];

			//loop through particles to find nearest neighbour
			for (i = bin_start_index; i < bin_start_index+bin_count; i++){
				float2 n_location = p->location[i];
				if (i != idx){ //cant be closest to itself
					//distance check (no need to square root)
					float n_dist_sq = (n_location.x - location.x)*(n_location.x - location.x) + (n_location.y - location.y)*(n_location.y - location.y);
					if (n_dist_sq < dist_sq){
						//we have found a new nearest neighbour if it is within the range
						if (n_dist_sq < INTERACTION_RANGE*INTERACTION_RANGE){
							dist_sq = n_dist_sq;
							nn = i;
						}
					}
				}
			}
		}
	}

	//write nearest neighbour
	p->nn_key[idx] = nn;
}

/* Thrust Implementation Additional Kernels */

__global__ void keyValues(particles *p, key_values *kv)
{
	//todo
}

//todo __global__ void reorderParticles(...)


//todo __global__ void histogramParticles(...)

__device__ __host__ int2 binLocation(float2 location){
	int bin_x = (int)(location.x / INTERACTION_RANGE);
	int bin_y = (int)(location.y / INTERACTION_RANGE);
	return make_int2(bin_x, bin_y);
}

__device__ __host__ int binIndex(int2 bin){
	return bin.x + bin.y*ENV_BIN_DIM;
}

/* Host Functions*/

int main(int argc, char **argv)
{
	particlesCPU();
	particlesGPU();

	return 0;
}

void particlesCPU()
{
	environment *h_env;
	environment *d_env;
	particles *h_particles;
	particles *h_particles_sorted;
	particles *d_particles;
	particles *d_particles_sorted;
	key_values *h_key_values;
	key_values *d_key_values;

	float time;
	clock_t begin, end;
	int errors;

	//allocate host memory (pinned)
	h_env = (environment*)malloc(sizeof(environment));
	h_particles = (particles*)malloc(sizeof(particles));
	h_particles_sorted = (particles*)malloc(sizeof(particles));
	h_key_values = (key_values*)malloc(sizeof(key_values));
	checkCUDAError("CPU version: Host malloc");

	//allocate device memory
	cudaMalloc((void**)&d_env, sizeof(environment));
	cudaMalloc((void**)&d_particles, sizeof(particles));
	cudaMalloc((void**)&d_particles_sorted, sizeof(particles));
	cudaMalloc((void**)&d_key_values, sizeof(key_values));
	checkCUDAError("CPU version: Device malloc");

	//set host data to 0
	memset(h_env, 0, sizeof(environment));
	memset(h_particles, 0, sizeof(particles));
	memset(h_key_values, 0, sizeof(key_values));

	//set device data to 0
	cudaMemset(d_env, 0, sizeof(environment));
	cudaMemset(d_particles, 0, sizeof(particles));
	cudaMemset(d_key_values, 0, sizeof(key_values));
	checkCUDAError("CPU version: Device memset");

	//init some particle data
	initParticles(h_particles);

	/* CPU implementation */
	cudaDeviceSynchronize();
	begin = clock();

	//key value pairs
	keyValuesCPU(h_particles, h_key_values);
	//sort particles on CPU
	sortKeyValuesCPU(h_key_values);
	//reorder particles
	reorderParticlesCPU(h_key_values, h_particles, h_particles_sorted);
	//histogram particle counts
	histogramParticlesCPU(h_particles_sorted, h_env);
	//prefix sum the environment bin locations
	prefixSumEnvironmentCPU(h_env);
	//host to device copy
	cudaMemcpy(d_particles_sorted, h_particles_sorted, sizeof(particles), cudaMemcpyHostToDevice);
	cudaMemcpy(d_env, h_env, sizeof(environment), cudaMemcpyHostToDevice);
	checkCUDAError("CPU version: Host 2 Device");
	//particle nearest neighbour kernel
	particleNNSearch <<<NUM_PARTICLES / TPB, TPB >>>(d_particles_sorted, d_env);
	checkCUDAError("CPU version: CPU version Kernel");
	//device to host copy
	cudaMemcpy(h_particles_sorted, d_particles_sorted, sizeof(particles), cudaMemcpyDeviceToHost);
	checkCUDAError("CPU version: Device 2 Host");

	//calculate timing
	cudaDeviceSynchronize();
	end = clock();
	time = (float)(end - begin) / CLOCKS_PER_SEC;

	errors = checkResults("CPU", h_particles_sorted);
	printf("CPU NN Search completed in %f seconds with %d errors\n", time, errors);

	//free host and device memory
	free(h_env);
	free(h_particles);
	free(h_particles_sorted);
	free(h_key_values);
	cudaFree(d_env);
	cudaFree(d_particles);
	cudaFree(d_particles_sorted);
	cudaFree(d_key_values);
	checkCUDAError("CPU version: CUDA free");

}


void particlesGPU()
{
	environment *h_env;
	environment *d_env;
	particles *h_particles;
	particles *h_particles_sorted;
	particles *d_particles;
	particles *d_particles_sorted;
	key_values *h_key_values;
	key_values *d_key_values;

	float time;
	clock_t begin, end;
	int errors;
	//allocate host memory (pinned)
	h_env = (environment*)malloc(sizeof(environment));
	h_particles = (particles*)malloc(sizeof(particles));
	h_particles_sorted = (particles*)malloc(sizeof(particles));
	h_key_values = (key_values*)malloc(sizeof(key_values));
	checkCUDAError("GPU version: Host malloc");

	//allocate device memory
	cudaMalloc((void**)&d_env, sizeof(environment));
	cudaMalloc((void**)&d_particles, sizeof(particles));
	cudaMalloc((void**)&d_particles_sorted, sizeof(particles));
	cudaMalloc((void**)&d_key_values, sizeof(key_values));
	checkCUDAError("GPU version: Device malloc");

	//set host data to 0
	memset(h_env, 0, sizeof(environment));
	memset(h_particles, 0, sizeof(particles));
	memset(h_key_values, 0, sizeof(key_values));

	//set device data to 0
	cudaMemset(d_env, 0, sizeof(environment));
	cudaMemset(d_particles, 0, sizeof(particles));
	cudaMemset(d_key_values, 0, sizeof(key_values));
	checkCUDAError("GPU version: Device memset");
	//init some particle data
	initParticles(h_particles);

	/* Thrust Implementation */
	cudaDeviceSynchronize();
	begin = clock();

	//Exercise 1.1) Copy from host to device
	//cudaMemcpy(...)
	checkCUDAError("GPU version: Host 2 Device");

	//Exercise 1.2) generate key value pairs on device
	keyValues << <NUM_PARTICLES / TPB, TPB >> >(d_particles, d_key_values);
	checkCUDAError("GPU version: Device keyValues");

	//Exercise 1.3) sort by key
	//thrust::sort_by_key(...);
	checkCUDAError("GPU version: Thrust sort");

	//Exercise 1.4) re-order
	//reorderParticles <<<...>>>
	checkCUDAError("GPU version: Device reorder");

	//Exercise 1.5) histogram
	//histogramParticles <<<...>>>(...);
	checkCUDAError("GPU version: Device Histogram");

	//Exercise 1.6) thrust prefix sum
	//exclusive_scan
	checkCUDAError("GPU version: Thrust scan");

	//particle nearest neighbour kernel
	particleNNSearch << <NUM_PARTICLES / TPB, TPB >> >(d_particles_sorted, d_env);
	checkCUDAError("GPU version: Kernel");

	//device to host copy
	cudaMemcpy(h_particles_sorted, d_particles_sorted, sizeof(particles), cudaMemcpyDeviceToHost);
	checkCUDAError("GPU version: Device 2 Host");

	//calculate timing
	cudaDeviceSynchronize();
	end = clock();
	time = (float)(end - begin) / CLOCKS_PER_SEC;

	/* print results and clean up*/
	errors = checkResults("GPU", h_particles_sorted);
	printf("GPU NN Search completed in %f seconds with %d errors\n", time, errors);

	
	//free host and device memory
	free(h_env);
	free(h_particles);
	free(h_particles_sorted);
	free(h_key_values);
	cudaFree(d_env);
	cudaFree(d_particles);
	cudaFree(d_particles_sorted);
	cudaFree(d_key_values);
	checkCUDAError("GPU version: CUDA free");

}




void initParticles(particles *p){
	//seed
	srand(123);
	
	//random positions
	for (int i = 0; i < NUM_PARTICLES; i++){
		float rand_x = rand() / (float)RAND_MAX * ENV_DIM;
		float rand_y = rand() / (float)RAND_MAX * ENV_DIM;
		float2 location = make_float2(rand_x, rand_y);
		p->location[i] = location;
	}
}


int checkResults(char* name, particles *p){
	int i, j, errors;

	errors = 0;

	for (i = 0; i < NUM_PARTICLES; i++){
		float2 location = p->location[i];
		float dist_sq = ENV_DIM*ENV_DIM;	//a big number
		int cpu_nn = -1;

		//find nearest neighbour on CPU
		for (j = 0; j < NUM_PARTICLES; j++){
			float2 n_location = p->location[j];
			if (j != i){ //cant be closest to itself
				//distance check (no need to square root)
				float n_dist_sq = (n_location.x - location.x)*(n_location.x - location.x) + (n_location.y - location.y)*(n_location.y - location.y);
				if (n_dist_sq < dist_sq){
					//we have found a new nearest neighbour if it is within the range
					if (n_dist_sq < INTERACTION_RANGE*INTERACTION_RANGE){
						dist_sq = n_dist_sq;
						cpu_nn = j;
					}
				}
			}
		}

		if (p->nn_key[i] != cpu_nn){
			fprintf(stderr, "Error: %s NN for index %d is %d, Ref NN is %u\n", name, i, p->nn_key[i], cpu_nn);
			errors++;
		}
	}


	return errors;
}

void keyValuesCPU(particles *p, key_values *kv){
	//random positions
	for (int i = 0; i < NUM_PARTICLES; i++){
		float2 location = p->location[i];
		kv->value[i] = i;
		kv->sorting_key[i] = binIndex(binLocation(location));
	}
}

void sortKeyValuesCPU(key_values *kv){
	int i, j;

	//simple (inefficient) CPU bubble sort
	for (i = 0; i < (NUM_PARTICLES - 1); i++)
	{
		for (j = 0; j < NUM_PARTICLES - i - 1; j++)
		{
			if (kv->sorting_key[j] > kv->sorting_key[j + 1])
			{
				//swap values
				int swap_key;
				int swap_sort_value;

				swap_key = kv->value[j];
				swap_sort_value = kv->sorting_key[j];

				kv->value[j] = kv->value[j + 1];
				kv->sorting_key[j] = kv->sorting_key[j + 1];

				kv->value[j + 1] = swap_key;
				kv->sorting_key[j + 1] = swap_sort_value;
			}
		}
	}
}

void reorderParticlesCPU(key_values *kv, particles *p, particles *p_sorted){
	int i;

	//re-order based on the old key
	for (i = 0; i < NUM_PARTICLES; i++){
		int old_index = kv->value[i];
		p_sorted->location[i] = p->location[old_index];
	}
}

void histogramParticlesCPU(particles *p, environment *env)
{
	int i;

	//loop through particles and increase the bin count for each environment bin
	for (i = 0; i < (NUM_PARTICLES - 1); i++)
	{
		int bin_location = binIndex(binLocation(p->location[i])); //recalculate the sort value
		env->count[bin_location]++;
	}
}

void prefixSumEnvironmentCPU(environment * env)
{
	int i;
	int sum = 0;

	//serial prefix sum
	for (i = 0; i < ENV_BINS; i++){
		env->start_index[i] = sum;
		sum += env->count[i];
	}
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
