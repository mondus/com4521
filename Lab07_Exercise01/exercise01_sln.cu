#include <stdlib.h>
#include <stdio.h>
#include <math.h>


// include kernels and cuda headers after definitions of structures
#include "kernels.cuh" 


void checkCUDAError(const char*);
void readRecords(student_record *records);
// Ex 1.1 (2/3), It is necessary to declare the function's prototype, before it's used in main()
// Otherwise, the function would need to be implemented before main()
void studentRecordAOS2SOA(student_record *aos, student_records *soa);

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

	// Ex 1.1 (1/3), Convert recordsAOS to a structure of arrays in h_records
	studentRecordAOS2SOA(recordsAOS, h_records);
	
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

// Ex 1.1 (3/3), This function simplies iterates the structures, and copies the data from aos to soa
void studentRecordAOS2SOA(student_record *aos, student_records *soa){
	for (int i = 0; i < NUM_RECORDS; i++){
		soa->student_ids[i] = aos[i].student_id;
		soa->assignment_marks[i] = aos[i].assignment_mark;
	}
}


void maximumMark_atomic(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	float max_mark;
	int max_mark_student_id;
	float time;
	cudaEvent_t start, stop;

	max_mark = 0.0f;
	max_mark_student_id = 0;

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

// Exercise 2)
void maximumMark_recursive(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	int i;
	float max_mark;
	int max_mark_student_id;
	student_records *d_records_temp;
	float time;
	cudaEvent_t start, stop;

	max_mark = 0.0f;
	max_mark_student_id = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("Recursive: CUDA memcpy");

	cudaEventRecord(start, 0);
	
	// Ex 2.3, Iteratively call GPU steps so that there are THREADS_PER_BLOCK values left
	for (i = NUM_RECORDS; i > THREADS_PER_BLOCK; i /= 2){

		dim3 blocksPerGrid(i / THREADS_PER_BLOCK, 1, 1);
		dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
		maximumMark_recursive_kernel << <blocksPerGrid, threadsPerBlock, sizeof(student_record)*THREADS_PER_BLOCK >> >(d_records, d_records_result);
		cudaDeviceSynchronize();
		checkCUDAError("Recursive: CUDA kernel");

		//swap input and output
		d_records_temp = d_records;
		d_records = d_records_result;
		d_records_result = d_records_temp;
	}

	// Ex 2.4, copy back the final THREADS_PER_BLOCK values
	cudaMemcpy(h_records_result->assignment_marks, d_records->assignment_marks, sizeof(float)*THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_records_result->student_ids, d_records->student_ids, sizeof(int)*THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);
	checkCUDAError("Recursive: CUDA memcpy back");

	// Ex 2.5, reduce the final THREADS_PER_BLOCK values on CPU
	max_mark = 0;
	max_mark_student_id = 0;
	for (i = 0; i < THREADS_PER_BLOCK; i++){
		float mark = h_records_result->assignment_marks[i];
		if (mark > max_mark){
			max_mark = mark;
			max_mark_student_id = h_records_result->student_ids[i];;
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	// output the result
	printf("Recursive: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// Ex 3)
void maximumMark_SM(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	unsigned int i;
	float max_mark;
	int max_mark_student_id;
	float time;
	cudaEvent_t start, stop;

	max_mark = 0.0f;
	max_mark_student_id = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("SM: CUDA memcpy");

	// Ex 3.4, Call the shared memory reduction kernel
	cudaEventRecord(start, 0);
	dim3 blocksPerGrid(NUM_RECORDS / THREADS_PER_BLOCK, 1, 1);
	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
	maximumMark_SM_kernel << <blocksPerGrid, threadsPerBlock, sizeof(student_record)*THREADS_PER_BLOCK >> >(d_records, d_records_result);
	cudaDeviceSynchronize();
	checkCUDAError("SM: CUDA kernel");


	// Ex 3.5, Copy the final block values back to CPU
	cudaMemcpy(h_records_result->assignment_marks, d_records_result->assignment_marks, sizeof(float)*blocksPerGrid.x, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_records_result->student_ids, d_records_result->student_ids, sizeof(int)*blocksPerGrid.x, cudaMemcpyDeviceToHost);
	checkCUDAError("SM: CUDA memcpy back");
	max_mark = 0;
	max_mark_student_id = 0;

	// Ex 3.6, Reduce the block level results on CPU
	for (i = 0; i < blocksPerGrid.x; i++){
		float mark = h_records_result->assignment_marks[i];
		if (mark > max_mark){
			max_mark = mark;
			max_mark_student_id = h_records_result->student_ids[i];;
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	// output result
	printf("SM: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// Exercise 4)
void maximumMark_shuffle(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	unsigned int i;
	unsigned int warps_per_grid;
	float max_mark;
	int max_mark_student_id;
	float time;
	cudaEvent_t start, stop;

	max_mark = 0.0f;
	max_mark_student_id = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("Shuffle: CUDA memcpy");

	// Ex 4.2, Execute the kernel
	cudaEventRecord(start, 0);
	dim3 blocksPerGrid(NUM_RECORDS / THREADS_PER_BLOCK, 1, 1);
	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
	warps_per_grid = NUM_RECORDS / 32;
	maximumMark_shuffle_kernel << <blocksPerGrid, threadsPerBlock >> >(d_records, d_records_result);
	cudaDeviceSynchronize();
	checkCUDAError("Shuffle: CUDA kernel");

	// copy the final warp values back to CPU
	cudaMemcpy(h_records_result->assignment_marks, d_records_result->assignment_marks, sizeof(float)*warps_per_grid, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_records_result->student_ids, d_records_result->student_ids, sizeof(int)*warps_per_grid, cudaMemcpyDeviceToHost);
	checkCUDAError("Shuffle: CUDA memcpy back");
	max_mark = 0;
	max_mark_student_id = 0;

	// reduce the warp level results on CPU
	for (i = 0; i < warps_per_grid; i++){
		float mark = h_records_result->assignment_marks[i];
		if (mark > max_mark){
			max_mark = mark;
			max_mark_student_id = h_records_result->student_ids[i];;
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	// output result
	printf("Shuffle: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}