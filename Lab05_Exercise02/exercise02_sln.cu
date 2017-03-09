/*
* Source code for this lab class is modifed from the book CUDA by Exmaple and provided by permission of NVIDIA Corporation
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define IMAGE_DIM 2048
#define SPHERE_SIZE_SAMPLES 8
#define STARTING_SPHERES 16
#define MAX_SPHERES STARTING_SPHERES<<(SPHERE_SIZE_SAMPLES -1)

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

void output_image_file(uchar4* image);
void checkCUDAError(const char *msg);

struct Sphere {
	float   r, b, g;
	float   radius;
	float   x, y, z;
};

/* Device Code */

__device__ float sphere_intersect(Sphere *s, float ox, float oy, float *n) {
	float dx = ox - s->x;
	float dy = oy - s->y;
	float radius = s->radius;
	if (dx*dx + dy*dy < radius*radius) {
		float dz = sqrtf(radius*radius - dx*dx - dy*dy);
		*n = dz / sqrtf(radius * radius);
		return dz + s->z;
	}
	return -INF;
}

__device__ float sphere_intersect_read_only(Sphere const* __restrict__ s, float ox, float oy, float *n) {
	float dx = ox - s->x;
	float dy = oy - s->y;
	float radius = s->radius;
	if (dx*dx + dy*dy < radius*radius) {
		float dz = sqrtf(radius*radius - dx*dx - dy*dy);
		*n = dz / sqrtf(radius * radius);
		return dz + s->z;
	}
	return -INF;
}

__constant__ Sphere d_const_s[MAX_SPHERES];
__constant__ unsigned int d_sphere_count;

__global__ void ray_trace(uchar4 *image, Sphere *d_s) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float   ox = (x - IMAGE_DIM / 2.0f);
	float   oy = (y - IMAGE_DIM / 2.0f);

	float   r = 0, g = 0, b = 0;
	float   maxz = -INF;
	for (int i = 0; i<d_sphere_count; i++) {
		Sphere *s = &d_s[i];
		float   n;
		float   t = sphere_intersect(s, ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s->r * fscale;
			g = s->g * fscale;
			b = s->b * fscale;
			maxz = t;
		}
	}

	image[offset].x = (int)(r * 255);
	image[offset].y = (int)(g * 255);
	image[offset].z = (int)(b * 255);
	image[offset].w = 255;
}

__global__ void ray_trace_const(uchar4 *image) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float   ox = (x - IMAGE_DIM / 2.0f);
	float   oy = (y - IMAGE_DIM / 2.0f);

	float   r = 0, g = 0, b = 0;
	float   maxz = -INF;
	for (int i = 0; i<d_sphere_count; i++) {
		Sphere *s = &d_const_s[i];
		float   n;
		float   t = sphere_intersect(s, ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s->r * fscale;
			g = s->g * fscale;
			b = s->b * fscale;
			maxz = t;
		}
	}

	image[offset].x = (int)(r * 255);
	image[offset].y = (int)(g * 255);
	image[offset].z = (int)(b * 255);
	image[offset].w = 255;
}

__global__ void ray_trace_read_only(uchar4 *image, Sphere const* __restrict__ d_s) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float   ox = (x - IMAGE_DIM / 2.0f);
	float   oy = (y - IMAGE_DIM / 2.0f);

	float   r = 0, g = 0, b = 0;
	float   maxz = -INF;
	for (int i = 0; i<d_sphere_count; i++) {
		Sphere const* __restrict__ s = &d_s[i];
		float   n;
		float   t = sphere_intersect_read_only(s, ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s->r * fscale;
			g = s->g * fscale;
			b = s->b * fscale;
			maxz = t;
		}
	}

	image[offset].x = (int)(r * 255);
	image[offset].y = (int)(g * 255);
	image[offset].z = (int)(b * 255);
	image[offset].w = 255;
}

/* Host code */

int main(void) {
	unsigned int image_size, spheres_size;
	uchar4 *d_image;
	uchar4 *h_image;
	cudaEvent_t     start, stop;
	Sphere h_s[MAX_SPHERES];
	Sphere *d_s;
	float3 timing_data[SPHERE_SIZE_SAMPLES]; //timing data for SPHERE_SIZE_SAMPLES sphere counts where [0]=normal, [1]=read-only, [2]=const

	image_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar4);
	spheres_size = sizeof(Sphere)*MAX_SPHERES;

	// create timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory on the GPU for the output image
	cudaMalloc((void**)&d_image, image_size);
	cudaMalloc((void**)&d_s, spheres_size);
	checkCUDAError("CUDA malloc");

	// create some random spheres
	for (int i = 0; i<MAX_SPHERES; i++) {
		h_s[i].r = rnd(1.0f);
		h_s[i].g = rnd(1.0f);
		h_s[i].b = rnd(1.0f);
		h_s[i].x = rnd((float)IMAGE_DIM) - (IMAGE_DIM / 2.0f);
		h_s[i].y = rnd((float)IMAGE_DIM) - (IMAGE_DIM / 2.0f);
		h_s[i].z = rnd((float)IMAGE_DIM) - (IMAGE_DIM / 2.0f);
		h_s[i].radius = rnd(100.0f) + 20;
	}
	//copy to constant memory
	cudaMemcpyToSymbol(d_const_s, h_s, spheres_size);
	//copy to device memory
	cudaMemcpy(d_s, h_s, spheres_size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");

	//generate host image
	h_image = (uchar4*)malloc(image_size);

	//cuda layout
	dim3    blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
	dim3    threadsPerBlock(16, 16);

	for (int i = 0; i < SPHERE_SIZE_SAMPLES; i++){
		unsigned int sphere_count = STARTING_SPHERES << i;
		printf("Executing code for sphere count %d\n", sphere_count);
		cudaMemcpyToSymbol(d_sphere_count, &sphere_count, sizeof(unsigned int));
		checkCUDAError("CUDA copy sphere count to device");

		// generate a image from the sphere data
		cudaEventRecord(start, 0);
		ray_trace << <blocksPerGrid, threadsPerBlock >> >(d_image, d_s);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timing_data[i].x, start, stop);
		checkCUDAError("kernel (normal)");

		// generate a image from the sphere data (using read only cache)
		cudaEventRecord(start, 0);
		ray_trace_read_only << <blocksPerGrid, threadsPerBlock >> >(d_image, d_s);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timing_data[i].y, start, stop);
		checkCUDAError("kernel (read-only)");

		// generate a image from the sphere data (using constant cache)
		cudaEventRecord(start, 0);
		ray_trace_const << <blocksPerGrid, threadsPerBlock >> >(d_image);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timing_data[i].z, start, stop);
		checkCUDAError("kernel (const)");
	}

	// copy the image back from the GPU for output to file
	cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	//output timings
	printf("Timing Data Table\n Spheres | Normal | Read-only | Const\n");
	for (int i = 0; i < SPHERE_SIZE_SAMPLES; i++){
		int sphere_count = STARTING_SPHERES << i;
		printf(" %-7i | %-6.3f | %-9.3f | %.3f\n", sphere_count, timing_data[i].x, timing_data[i].y, timing_data[i].z);
	}

	// output image
	output_image_file(h_image);

	//cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_image);
	cudaFree(d_s);
	free(h_image);

	return 0;
}

void output_image_file(uchar4* image)
{
	FILE *f; //output file handle

	//open the output file and write header info for PPM filetype
	f = fopen("output.ppm", "wb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "# COM4521 Lab 05 Exercise02\n");
	fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
	for (int x = 0; x < IMAGE_DIM; x++){
		for (int y = 0; y < IMAGE_DIM; y++){
			int i = x + y*IMAGE_DIM;
			fwrite(&image[i], sizeof(unsigned char), 3, f); //only write rgb (ignoring a)
		}
	}

	fclose(f);
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
