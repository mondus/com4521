#ifndef KERNEL_H //ensures header is only included once
#define KERNEL_H

//#ifndef __CUDACC__
//#define __CUDACC__
//#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define NUM_RECORDS 2048
#define THREADS_PER_BLOCK 256
#define SQRT_THREADS_PER_BLOCK sqrt(THREADS_PER_BLOCK)

struct student_record{
	int student_id;
	float assignment_mark;
};

struct student_records{
	int student_ids[NUM_RECORDS];
	float assignment_marks[NUM_RECORDS];
};

typedef struct student_record student_record;
typedef struct student_records student_records;

__device__ float d_max_mark = 0;
__device__ int d_max_mark_student_id = 0;

// lock for global Atomics
#define UNLOCKED 0
#define LOCKED   1
__device__ volatile int lock = UNLOCKED;

// Function creates an atomic compare and swap to save the maximum mark and associated student id
__device__ void setMaxMarkAtomic(float mark, int id) {
	bool needlock = true;

	while (needlock){
		// get lock to perform critical section of code
		if (atomicCAS((int *)&lock, UNLOCKED, LOCKED) == 0){

			//critical section of code
			if (d_max_mark < mark){
				d_max_mark_student_id = id;
				d_max_mark = mark;
			}

			// free lock
			atomicExch((int*)&lock, 0);
			needlock = false;
			
			__threadfence(); // gurentees that other threads are have completed global memory writes
		}
	}
}

// Naive atomic implementation
__global__ void maximumMark_atomic_kernel(student_records *d_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float mark = d_records->assignment_marks[idx];
	int id = d_records->student_ids[idx];

	setMaxMarkAtomic(mark, id);

}

// Exercise 2) Recursive Reduction
// This kernel is executed multiple times, each time halving the number of records to be processed
__global__ void maximumMark_recursive_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// The size of this shared memory is dynamic and specified at kernel launch!
	extern __shared__ student_record sdata[];

	// Ex 2.1, Load a single student record into shared memory
	// Remember to call __syncthreads() after changing shared memory, so all threads in the block see the change
	sdata[threadIdx.x].assignment_mark = d_records->assignment_marks[idx];
	sdata[threadIdx.x].student_id = d_records->student_ids[idx];
	__syncthreads();

	// Ex 2.2, Compare two values and write the result to d_reduced_records
	// Every even indexed thread accesses two values, and returns the minimum to global memory
	// Therefore there is no potential for a race condition here
	if (idx % 2 == 0){
		if (sdata[threadIdx.x].assignment_mark < sdata[threadIdx.x + 1].assignment_mark){
			sdata[threadIdx.x] = sdata[threadIdx.x + 1];
		}

		// write result
		// Ensure the output is compact, such by using the index idx/2 (as odd threads are not writing)
		d_reduced_records->assignment_marks[idx / 2] = sdata[threadIdx.x].assignment_mark;
		d_reduced_records->student_ids[idx / 2] = sdata[threadIdx.x].student_id;
	}


}


// Exercise 3) Using block level reduction
__global__ void maximumMark_SM_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ student_record sdata[];

	// Ex 3.1, Load a single student record into shared memory
	// Remember to call __syncthreads() after changing shared memory, so all threads in the block see the change
	sdata[threadIdx.x].assignment_mark = d_records->assignment_marks[idx];
	sdata[threadIdx.x].student_id = d_records->student_ids[idx];
	__syncthreads();


	// Ex 3.2, Strided shared memory conflict free reduction
	// Different threads will access the same indices as other threads with each iteration, so __syncthreads() is required!
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1){
		if (threadIdx.x < stride){
			if (sdata[threadIdx.x].assignment_mark < sdata[threadIdx.x + stride].assignment_mark){
				sdata[threadIdx.x] = sdata[threadIdx.x + stride];
			}
		}
		__syncthreads();
	}


	// Ex 3.3, Write the result
	// Only the first thread of each block needs to output a result
	if (threadIdx.x == 0){
		d_reduced_records->assignment_marks[blockIdx.x] = sdata[0].assignment_mark;
		d_reduced_records->student_ids[blockIdx.x] = sdata[0].student_id;
	}
}

// Exercise 4) Using warp level reduction
__global__ void maximumMark_shuffle_kernel(student_records *d_records, student_records *d_reduced_records) {
	// Ex 4.1, Complete the kernel
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// load a single record into local variable (registers)
	float assignment_mark = d_records->assignment_marks[idx];
	int student_id = d_records->student_ids[idx];

	// shuffle down
	// offset >>= 1, performs a single bit shift, essentially dividing by two. offset will go through 16, 8, 4, 2, 0
	for (int offset = 16; offset > 0; offset >>= 1){
		// _shfl_down() has implicit warp synchronisation, so __syncthreads() is not required!
		float shuffle_mark = __shfl_down(assignment_mark, offset);
		int shuffle_id = __shfl_down(student_id, offset);
		if (assignment_mark < shuffle_mark){
			assignment_mark = shuffle_mark;
			student_id = shuffle_id;
		}
	}

	// write result if first thread in warp (every 32nd thread)
	if (threadIdx.x % 32 == 0){
		unsigned int o_index = idx / 32;
		d_reduced_records->assignment_marks[o_index] = assignment_mark;
		d_reduced_records->student_ids[o_index] = student_id;
	}
}

#endif //KERNEL_H
