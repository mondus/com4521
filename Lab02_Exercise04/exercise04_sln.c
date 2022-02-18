
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 1024

typedef double matrix_type;

typedef matrix_type **matrixNN;

void init_random_matrix(matrixNN m);

void init_empty_matrix(matrixNN m);

void write_matrix_to_file(const char *filename, const matrixNN r);


void multiply_A(matrixNN r, const matrixNN a, const matrixNN b);

void multiply_B(matrixNN r, const matrixNN a, const matrixNN b);

void transpose(matrixNN t);

void multiply_C(matrixNN r, const matrixNN a, const matrixNN b);

void multiply_C_unrolled(matrixNN r, const matrixNN a, const matrixNN t);

void main(){
	clock_t begin, end;
	double seconds;
	matrixNN a;
	matrixNN b;
	matrixNN c;
	int i;



	a = (matrixNN)malloc(sizeof(matrix_type*)*N);
	for (i = 0; i < N;i++)
		a[i] = (matrix_type*)malloc(sizeof(matrix_type)*N);
	b = (matrixNN)malloc(sizeof(matrix_type*)*N);
	for (i = 0; i < N; i++)
		b[i] = (matrix_type*)malloc(sizeof(matrix_type)*N);
	c = (matrixNN)malloc(sizeof(matrix_type*)*N);
	for (i = 0; i < N; i++)
		c[i] = (matrix_type*)malloc(sizeof(matrix_type)*N);

	init_random_matrix(a);
	init_random_matrix(b);
	init_empty_matrix(c);

	begin = clock();
	transpose(b);
	multiply_C(c, a, b);

	end = clock();
	seconds = (end - begin) / (double)CLOCKS_PER_SEC;


	printf("Matrix multiply complete in %.2f seconds\n", seconds);

	write_matrix_to_file("matrix_mul.txt", c);

	printf("Done writing results\n");

	for (i = 0; i < N; i++)
		free(a[i]);
	free(a);
	for (i = 0; i < N; i++)
		free(b[i]);
	free(b);
	for (i = 0; i < N; i++)
		free(c[i]);
	free(c);
}



void init_random_matrix(matrixNN m){
	int i, j;
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
				//m[i][j] = rand() % 100; //for integer
				m[i][j] = rand() / (matrix_type)RAND_MAX; //for double and float
		}
	}
}

void init_empty_matrix(matrixNN m){
	int i, j;
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			m[i][j] = 0;
		}
	}
}


void multiply_A(matrixNN r, const matrixNN a, const matrixNN b){
	int i, j, k;

	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			r[i][j] = 0;
			for (k = 0; k < N; k++){
				r[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

// Ex 4.1, The variable temp is declared at local scope, and used to calculate the sum
// Local variables can be preferable, as they are more likely to reside in low latency caches/memory
// The impact of this will be more visible when we later use CUDA
void multiply_B(matrixNN r, const matrixNN a, const matrixNN b){
	int i, j, k;
	matrix_type temp;

	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			temp = 0;
			for (k = 0; k < N; k++){
				temp += a[i][k] * b[k][j];
			}
			r[i][j] = temp;
		}
	}
}

// Ex 4.3, (1/2) Transposing the matrix allows memory access to be coalesced (neighbouring) during the multiply function
// When a processor accesses pulls data from RAM into processor caches, it accesses cache lines worth of data at once
// e.g. 32-128 bytes, hence reading along cache lines, as opposed to scattered accesses, can reduce the calls to higher latency RAM
// Caches are only small, so scattered accesses may cause earlier spare data from cache lines to be evicted before it is required.
void transpose(matrixNN t){
	int i, j;
	matrix_type temp;

	for (i = 0; i < N; i++){
		for (j = i+1; j < N; j++){
			temp = t[i][j];
			t[i][j] = t[j][i];
			t[j][i] = temp;
		}
	}

}

// Ex 4.3, (2/2) Now that matrix b has been transposed it is accessed [j][k] rather than [k][j] (see multiply_B())
void multiply_C(matrixNN r, const matrixNN a, const matrixNN b){
	int i, j, k;
	matrix_type temp;

	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			temp = 0;
			for (k = 0; k < N; k++){
				temp += a[i][k] * b[j][k];
			}
			r[i][j] = temp;
		}
	}
}


// Ex 4.(4), Loop unrolling is the process of manually writing the code which would normally be iterated.
// Normally a compiler will automatically unroll suitable (e.g. fixed length) loops
// However, in some cases manual unrolling could be required if the compiler is unable to perform it automatically
// Loop unrolling removes the runtime cost of loop maintenance, (e.g. incrementing i, j, k), by performing the computations at compile time
void multiply_C_unrolled(matrixNN r, const matrixNN a, const matrixNN t){
	int i, j, k;
	matrix_type temp;

	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			temp = 0;
			for (k = 0; k < N; k += 8){
				temp += a[i][k] * t[j][k];
				temp += a[i][k + 1] * t[j][k + 1];
				temp += a[i][k + 2] * t[j][k + 2];
				temp += a[i][k + 3] * t[j][k + 3];
				temp += a[i][k + 4] * t[j][k + 4];
				temp += a[i][k + 5] * t[j][k + 5];
				temp += a[i][k + 6] * t[j][k + 6];
				temp += a[i][k + 7] * t[j][k + 7];
			}
			r[i][j] = temp;
		}
	}
}


void write_matrix_to_file(const char *filename, const matrixNN r){
	FILE *f;
	int i, j;

	f = fopen(filename, "w");
	if (f == NULL){
		fprintf(stderr, "Error opening file '%s' for write\n", filename);
		return;
	}
	
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			fprintf(f, "%0.2f\t", r[i][j]);
		}
		fprintf(f, "\n");
	}

}
