#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef enum {
	CALCULATOR_ADD,
	CALCULATOR_SUB,
	CALCULATOR_DIV,
	CALCULATOR_MUL
} CALCULATOR_COMMANDS;

typedef enum{
	INPUT_RANDOM,
	INPUT_LINEAR
}INPUT_TYPE;

#define SAMPLES 262144
#define TPB 256
#define NUM_STREAMS 2
#define FILE_BUFFER_SIZE 32
#define MAX_COMMANDS 32
#define INPUT INPUT_LINEAR

__constant__ CALCULATOR_COMMANDS d_commands[MAX_COMMANDS];
__constant__ float d_operands[MAX_COMMANDS];

int readCommandsFromFile(CALCULATOR_COMMANDS *commands, float *operands);
void initInput(float *input);
void checkCUDAError(const char *msg);
int readLine(FILE *f, char buffer[]);
void cudaCalculatorDefaultStream(CALCULATOR_COMMANDS *commands, float *operands, int num_commands);
void cudaCalculatorNStream1(CALCULATOR_COMMANDS *commands, float *operands, int num_commands);
void cudaCalculatorNStream2(CALCULATOR_COMMANDS *commands, float *operands, int num_commands);
int checkResults(float* h_input, float* h_output, CALCULATOR_COMMANDS *commands, float *operands, int num_commands);

__global__ void parallelCalculator(float *input, float *output, int num_commands)
{
	float out;
	int idx;

	idx = threadIdx.x + blockIdx.x*blockDim.x;

	//get input
	out = input[idx];

	//apply commands
	for (int i = 0; i < num_commands; i++){
		CALCULATOR_COMMANDS cmd = d_commands[i];
		float v = d_operands[i];

		switch (cmd){
		case(CALCULATOR_ADD) : {
								   out += v;
								   break;
		}
		case(CALCULATOR_SUB) : {
								   out -= v;
								   break;
		}
		case(CALCULATOR_DIV) : {
								   out /= v;
								   break;
		}
		case(CALCULATOR_MUL) : {
								   out *= v;
								   break;
		}
		}
	}

	output[idx] = out;
}


int main(int argc, char**argv){
	int num_commands;

	CALCULATOR_COMMANDS h_commands[MAX_COMMANDS];
	float h_operands[MAX_COMMANDS];

	//get calculator operators from file
	num_commands = readCommandsFromFile(h_commands, h_operands);

	printf("%d commands found in file\n", num_commands);

	//copy commands and operands to device
	cudaMemcpyToSymbol(d_commands, h_commands, sizeof(CALCULATOR_COMMANDS)*MAX_COMMANDS);
	checkCUDAError("Commands copy to constant memory");
	cudaMemcpyToSymbol(d_operands, h_operands, sizeof(float)*MAX_COMMANDS);
	checkCUDAError("Commands copy to constant memory");

	//perform fully synchronous version
	cudaCalculatorDefaultStream(h_commands, h_operands, num_commands);

	//perform asynchronous version
	cudaCalculatorNStream1(h_commands, h_operands, num_commands);

	//perform asynchronous version
	cudaCalculatorNStream2(h_commands, h_operands, num_commands);


}

void cudaCalculatorDefaultStream(CALCULATOR_COMMANDS *commands, float *operands, int num_commands){
	float *h_input, *h_output;
	float *d_input, *d_output;
	float time;
	cudaEvent_t start, stop;
	int errors;

	//init cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//allocate memory
	h_input = (float*)malloc(sizeof(float)*SAMPLES);
	h_output = (float*)malloc(sizeof(float)*SAMPLES);

	//allocate device memory
	cudaMalloc((void**)&d_input, sizeof(float)*SAMPLES);
	cudaMalloc((void**)&d_output, sizeof(float)*SAMPLES);
	checkCUDAError("CUDA Memory allocate: default stream");

	//init the host input
	initInput(h_input);

	//begin timing
	cudaEventRecord(start);

	//1) Asynchronous host to device memory copy
	cudaMemcpy(d_input, h_input, sizeof(float)*SAMPLES, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA Memory copy H2D: default stream");


	//2) Execute kernel
	parallelCalculator << <SAMPLES / TPB, TPB >> >(d_input, d_output, num_commands);
	checkCUDAError("CUDA Kernel: default stream");

	//3) Asynchronousdevice to host memory copy
	cudaMemcpy(h_output, d_output, sizeof(float)*SAMPLES, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA Memory copy D2H: default stream");

	//end timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//check for errors and print timing
	errors = checkResults(h_input, h_output, commands, operands, num_commands);
	printf("Synchronous V Completed in %f seconds with %d errors\n", time, errors);

	//cleanup
	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);
	free(h_output);
}

void cudaCalculatorNStream1(CALCULATOR_COMMANDS *commands, float *operands, int num_commands){
	float *h_input, *h_output;
	float *d_input, *d_output;
	float time;
	cudaEvent_t start, stop;
	int i, errors;
	cudaStream_t streams[NUM_STREAMS];

	//init cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//allocate memory
	cudaMallocHost((void**)&h_input, sizeof(float)*SAMPLES);
	cudaMallocHost((void**)&h_output, sizeof(float)*SAMPLES);

	//allocate device memory
	cudaMalloc((void**)&d_input, sizeof(float)*SAMPLES);
	cudaMalloc((void**)&d_output, sizeof(float)*SAMPLES);
	checkCUDAError("CUDA Memory allocate: default stream");

	//init streams
	for (i = 0; i < NUM_STREAMS; i++){
		//create the stream
		cudaStreamCreate(&streams[i]);
	}

	//init the host input
	initInput(h_input);

	//begin timing
	cudaEventRecord(start);

	int batch_samples = SAMPLES / NUM_STREAMS;
	for (i = 0; i < NUM_STREAMS; i++){
		int offset = i*batch_samples;

		//1) Asynchronous host to device memory copy
		cudaMemcpyAsync(d_input + offset, h_input + offset, sizeof(float)*batch_samples, cudaMemcpyHostToDevice, streams[i]);
		checkCUDAError("CUDA Memory copy H2D: N streams");

		//2) Execute kernel
		parallelCalculator << <batch_samples / TPB, TPB, 0, streams[i] >> >(d_input + offset, d_output + offset, num_commands);
		checkCUDAError("CUDA Kernel: N streams");

		//3) Asynchronous device to host memory copy
		cudaMemcpyAsync(h_output + offset, d_output + offset, sizeof(float)*batch_samples, cudaMemcpyDeviceToHost, streams[i]);
		checkCUDAError("CUDA Memory copy D2H: N streams");
	}

	//end timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//check for errors and print timing
	errors = checkResults(h_input, h_output, commands, operands, num_commands);
	printf("Async V1 (%d streams) Completed in %f seconds with %d errors\n", NUM_STREAMS, time, errors);

	//cleanup
	//init streams
	for (i = 0; i < NUM_STREAMS; i++){
		//create the stream
		cudaStreamDestroy(streams[i]);
	}
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFreeHost(h_input);
	cudaFreeHost(h_output);
}


void cudaCalculatorNStream2(CALCULATOR_COMMANDS *commands, float *operands, int num_commands){
	float *h_input, *h_output;
	float *d_input, *d_output;
	float time;
	cudaEvent_t start, stop;
	int i, errors;
	cudaStream_t streams[NUM_STREAMS];

	//init cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//allocate memory
	cudaMallocHost((void**)&h_input, sizeof(float)*SAMPLES);
	cudaMallocHost((void**)&h_output, sizeof(float)*SAMPLES);

	//allocate device memory
	cudaMalloc((void**)&d_input, sizeof(float)*SAMPLES);
	cudaMalloc((void**)&d_output, sizeof(float)*SAMPLES);
	checkCUDAError("CUDA Memory allocate: default stream");

	//init streams
	for (i = 0; i < NUM_STREAMS; i++){
		//create the stream
		cudaStreamCreate(&streams[i]);
	}

	//init the host input
	initInput(h_input);

	//begin timing
	cudaEventRecord(start);

	int batch_samples = SAMPLES / NUM_STREAMS;

	for (i = 0; i < NUM_STREAMS; i++){
		int offset = i*batch_samples;
		//1) Asynchronous host to device memory copy
		cudaMemcpyAsync(d_input + offset, h_input + offset, sizeof(float)*batch_samples, cudaMemcpyHostToDevice, streams[i]);
		checkCUDAError("CUDA Memory copy H2D: N streams");
	}

	for (i = 0; i < NUM_STREAMS; i++){
		int offset = i*batch_samples;
		//2) Execute kernel
		parallelCalculator << <batch_samples / TPB, TPB, 0, streams[i] >> >(d_input + offset, d_output + offset, num_commands);
		checkCUDAError("CUDA Kernel: N streams");

	}

	for (i = 0; i < NUM_STREAMS; i++){
		int offset = i*batch_samples;
		//3) Asynchronous device to host memory copy
		cudaMemcpyAsync(h_output + offset, d_output + offset, sizeof(float)*batch_samples, cudaMemcpyDeviceToHost, streams[i]);
		checkCUDAError("CUDA Memory copy D2H: N streams");
	}


	//end timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//check for errors and print timing
	errors = checkResults(h_input, h_output, commands, operands, num_commands);
	printf("Async V2 (%d streams) Completed in %f seconds with %d errors\n", NUM_STREAMS, time, errors);

	//cleanup
	//init streams
	for (i = 0; i < NUM_STREAMS; i++){
		//create the stream
		cudaStreamDestroy(streams[i]);
	}
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFreeHost(h_input);
	cudaFreeHost(h_output);
}

int readCommandsFromFile(CALCULATOR_COMMANDS *commands, float *operands)
{
	FILE *f;
	float in_value;
	unsigned int line;
	char buffer[FILE_BUFFER_SIZE];
	char command[4];
	line = 0;

	printf("Welcome to the COM4521 Parallel floating point Calculator\n");
	f = fopen("commands.calc", "r");
	if (f == NULL){
		fprintf(stderr, "File not found\n");
		return 0;
	}


	while (readLine(f, buffer)){
		line++;

		if (line >= MAX_COMMANDS){
			fprintf(stderr, "To many commands in form maximum is %u\n", MAX_COMMANDS);
			return 0;
		}

		if (!(isalpha(buffer[0]) && isalpha(buffer[1]) && isalpha(buffer[2]) && buffer[3] == ' ')){
			fprintf(stderr, "Incorrect command format at line %u\n", line);
			return 0;
		}

		sscanf(buffer, "%s %f", command, &in_value);

		if (strcmp(command, "add") == 0){
			commands[line] = CALCULATOR_ADD;
		}
		else if (strcmp(command, "sub") == 0){
			commands[line] = CALCULATOR_SUB;
		}
		else if (strcmp(command, "div") == 0){
			commands[line] = CALCULATOR_DIV;
		}
		else if (strcmp(command, "mul") == 0){
			commands[line] = CALCULATOR_MUL;
		}
		else{
			fprintf(stderr, "Unknown command at line %u!\n", line);
			return 0;
		}

		operands[line] = in_value;

	}

	fclose(f);

	return line;
}


void initInput(float *input){
	int i;

	for (i = 0; i < SAMPLES; i++){
		if (INPUT == INPUT_LINEAR)
			input[i] = (float)i;
		else if (INPUT == INPUT_RANDOM)
			input[i] = rand() / (float)RAND_MAX;
	}
}

int readLine(FILE *f, char buffer[]){
	int i = 0;
	char c;
	while ((c = getc(f)) != '\n'){
		if (c == EOF)
			return 0;
		buffer[i++] = c;
		if (i == FILE_BUFFER_SIZE){
			fprintf(stderr, "Buffer size is too small for line input\n");
			exit(0);
		}
	}
	buffer[i] = '\0';

	if (strncmp(buffer, "exit", 4) == 0)
		return 0;
	else
		return 1;

}

int checkResults(float* h_input, float* h_output, CALCULATOR_COMMANDS *commands, float *operands, int num_commands)
{
	int i, j, errors;

	errors = 0;
	for (i = 0; i < SAMPLES; i++){
		float out = h_input[i];
		for (j = 0; j < num_commands; j++){
			CALCULATOR_COMMANDS cmd = commands[j];
			float v = operands[j];

			switch (cmd){
			case(CALCULATOR_ADD) : {
									   out += v;
									   break;
			}
			case(CALCULATOR_SUB) : {
									   out -= v;
									   break;
			}
			case(CALCULATOR_DIV) : {
									   out /= v;
									   break;
			}
			case(CALCULATOR_MUL) : {
									   out *= v;
									   break;
			}
			}
		}
		//test the result
		if (h_output[i] != out){
			//fprintf(stderr, "Error: GPU result (%f) differs from CPU result (%f) at index %d\n", h_output[i], out, i);
			errors++;
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