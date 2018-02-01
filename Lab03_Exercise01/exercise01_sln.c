
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N 1024

typedef double matrix_type;

typedef matrix_type **matrixNN;

void init_random_matrix(matrixNN m);

void init_empty_matrix(matrixNN m);

void write_matrix_to_file(const char *filename, const matrixNN r);

void transpose(matrixNN t);

void multiply(matrixNN r, const matrixNN a, const matrixNN b);


void main(){
	double begin, end;
	double seconds;
	matrixNN a;
	matrixNN b;
	matrixNN c;
	int i;

	int max_threads = omp_get_max_threads();
        printf("OpenMP using %d threads\n", max_threads);



	a = (matrixNN)malloc(sizeof(matrix_type)*N);
	for (i = 0; i < N; i++)
		a[i] = (matrix_type*)malloc(sizeof(matrix_type)*N);
	b = (matrixNN)malloc(sizeof(matrix_type)*N);
	for (i = 0; i < N; i++)
		b[i] = (matrix_type*)malloc(sizeof(matrix_type)*N);
	c = (matrixNN)malloc(sizeof(matrix_type)*N);
	for (i = 0; i < N; i++)
		c[i] = (matrix_type*)malloc(sizeof(matrix_type)*N);

	init_random_matrix(a);
	init_random_matrix(b);
	init_empty_matrix(c);

	begin = omp_get_wtime();
	transpose(b);
	multiply(c, a, b);

	end = omp_get_wtime();
	seconds = (end - begin);


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


void transpose(matrixNN t){
	int i, j;
	matrix_type temp;

	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			temp = t[i][j];
			t[i][j] = t[j][i];
			t[j][i] = temp;
		}
	}

}


void multiply(matrixNN r, const matrixNN a, const matrixNN t){
	int i, j, k;
	matrix_type temp;
// a and t are const so are implicitly shared variables
#pragma omp parallel for shared(r) private(i, j, k, temp) 
		for (i = 0; i < N; i++){
			for (j = 0; j < N; j++){
				temp = 0;
				for (k = 0; k < N; k++){
					temp += a[i][k] * t[j][k];
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
