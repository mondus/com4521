#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// include kernels and cuda headers after definitions of structures
#include "kernels.cuh" 


void checkCUDAError(const char*);
void readRecords(student_record *records);

void maximumMark_atomic(student_records*, student_records*, student_records*, student_records*);
void maximumMark_recursive(student_records*, student_records*, student_records*, student_records*);
void maximumMark_SM(student_records*, student_records*, student_records*, student_records*);
void maximumMark_shuffle(student_records*, student_records*, student_records*, student_records*);


int main(void) {
	student_record *recordsAOS;
	student_records *h_records;
	student_records *h_records_result;
	student_records *d_records;
	student_records *d_records_result;
	
	//host allocation
	recordsAOS = (student_record*)malloc(sizeof(student_record)*NUM_RECORDS);
	h_records = (student_records*)malloc(sizeof(student_records));
	h_records_result = (student_records*)malloc(sizeof(student_records));

	//device allocation
	cudaMalloc((void**)&d_records, sizeof(student_records));
	cudaMalloc((void**)&d_records_result, sizeof(student_records));
	checkCUDAError("CUDA malloc");

	//read file
	readRecords(recordsAOS);

	//Exercise 1.1) Convert recordsAOS to a structure of arrays in h_records
	
	//free AOS as it is no longer needed
	free(recordsAOS);

	//apply each approach in turn 
	maximumMark_atomic(h_records, h_records_result, d_records, d_records_result);
	maximumMark_recursive(h_records, h_records_result, d_records, d_records_result);
	maximumMark_SM(h_records, h_records_result, d_records, d_records_result);
	maximumMark_shuffle(h_records, h_records_result, d_records, d_records_result);


	// Cleanup
	free(h_records);
	free(h_records_result);
	cudaFree(d_records);
	cudaFree(d_records_result);
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

void readRecords(student_record *records){
	FILE *f = NULL;
	f = fopen("com4521.dat", "rb"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find com4521.dat file \n");
		exit(1);
	}

	//read student data
	if (fread(records, sizeof(student_record), NUM_RECORDS, f) != NUM_RECORDS){
		fprintf(stderr, "Error: Unexpected end of file!\n");
		exit(1);
	}
	fclose(f);
}


void maximumMark_atomic(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	float max_mark;
	int max_mark_student_id;
	float time;
	cudaEvent_t start, stop;
	
	max_mark = 0;
	max_mark_student_id = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("1) CUDA memcpy");

	cudaEventRecord(start, 0);
	//find highest mark using GPU
	dim3 blocksPerGrid(NUM_RECORDS / THREADS_PER_BLOCK, 1, 1);
	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
	maximumMark_atomic_kernel << <blocksPerGrid, threadsPerBlock >> >(d_records);
	cudaDeviceSynchronize();
	checkCUDAError("Atomics: CUDA kernel");

	// Copy result back to host
	cudaMemcpyFromSymbol(&max_mark, d_max_mark, sizeof(float));
	cudaMemcpyFromSymbol(&max_mark_student_id, d_max_mark_student_id, sizeof(int));
	checkCUDAError("Atomics: CUDA memcpy back");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//output result
	printf("Atomics: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

//Exercise 2)
void maximumMark_recursive(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	int i;
	float max_mark;
	int max_mark_student_id;
	student_records *d_records_temp;
	float time;
	cudaEvent_t start, stop;
	
	max_mark = 0;
	max_mark_student_id = 0.0f;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("Recursive: CUDA memcpy");

	cudaEventRecord(start, 0);
	
	//Exercise 2.3) Recursively call GPU steps until there are THREADS_PER_BLOCK values left

	//Exercise 2.4) copy back the final THREADS_PER_BLOCK values

	//Exercise 2.5) reduce the final THREADS_PER_BLOCK values on CPU


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//output the result
	printf("Recursive: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

//Exercise 3)
void maximumMark_SM(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	unsigned int i;
	float max_mark;
	int max_mark_student_id;
	float time;
	cudaEvent_t start, stop;
	
	max_mark = 0;
	max_mark_student_id = 0.0f;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("SM: CUDA memcpy");

	cudaEventRecord(start, 0);
	
	//Exercise 3.4) Call the shared memory reduction kernel

	//Exercise 3.5) Copy the final block values back to CPU

	//Exercise 3.6) Reduce the block level results on CPU

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//output result
	printf("SM: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

//Exercise 4)
void maximumMark_shuffle(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	unsigned int i;
	unsigned int warps_per_grid;
	float max_mark;
	int max_mark_student_id;
	float time;
	cudaEvent_t start, stop;
	
	max_mark = 0;
	max_mark_student_id = 0.0f;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("Shuffle: CUDA memcpy");
	
	cudaEventRecord(start, 0);

	//Exercise 4.2) Execute the kernel, copy back result, reduce final values on CPU

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//output result
	printf("Shuffle: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}