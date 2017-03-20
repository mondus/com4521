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

//Exercise 2) Recursive Reduction
__global__ void maximumMark_recursive_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Exercise 2.1) Load a single student record into shared memory

	//Exercise 2.2) Compare two values and write the result to d_reduced_records

}


//Exercise 3) Using block level reduction
__global__ void maximumMark_SM_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Exercise 3.1) Load a single student record into shared memory

	//Exercise 3.2) Strided shared memory conflict free reduction

	//Exercise 3.3) Write the result

}

//Exercise 4) Using warp level reduction
__global__ void maximumMark_shuffle_kernel(student_records *d_records, student_records *d_reduced_records) {
	//Exercise 4.1) Complete the kernel
	
}

#endif //KERNEL_H