#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "mandelbrot.h"

//image size
#define WIDTH 1024
#define HEIGHT 768

#define MAX_ITERATIONS 18				//number of iterations

//C parameters (modify these to change the zoom and position of the mandelbrot)
#define ZOOM 1.0
#define X_DISPLACEMENT -0.5
#define Y_DISPLACEMENT 0.0



static int iterations[HEIGHT][WIDTH];					//store the escape time (iteration count) as an integer
static double iterations_d[HEIGHT][WIDTH];				//store for the escape time as a double (with fractional part) for NIC method only
static int histogram[MAX_ITERATIONS + 1];					//histogram if escape times
static rgb rand_banding[MAX_ITERATIONS + 1];			//random colour banding
static int local_histogram[HEIGHT][MAX_ITERATIONS + 1];	//only required for exercise 2.2.3, HEIGHT is the maximum possible number of threads that could be initialised as it the maximum size of the number of work units (width of parallel loop)
static rgb rgb_output[HEIGHT][WIDTH];					//output data

const TRANSFER_FUNCTION tf = RANDOM_NORMALISED_ITERATION_COUNT;
const HISTOGRAM_METHOD hist_method = OMP_ATOMIC;

int main(int argc, char *argv[])
{
	int i, x, y;										//loop counters
	double c_r, c_i;									//real and imaginary part of the constant c
	double n_r, n_i, o_r, o_i;							//real and imaginary parts of new and old z
	double mu;											//iteration with fractional component
	double begin, end;									//timers
	double elapsed;										//elapsed time
	FILE *f;											//output file handle


	//open the output file and write header info for PPM filetype
	f = fopen("output.ppm", "wb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "# COM4521 Lab 03 Exercise02\n");
	fprintf(f, "%d %d\n%d\n", WIDTH, HEIGHT, 255);

	//start timer
	begin = omp_get_wtime();

	int temp = sizeof(local_histogram);
	//clear the histogram initial values
	memset(histogram, 0, sizeof(histogram));
	if (hist_method == OMP_MASTER){
		memset(local_histogram, 0, sizeof(local_histogram));
	}

	//random colour banding
	for (i = 0; i < MAX_ITERATIONS; i++){
		rand_banding[i].r = rand() % 128;
		rand_banding[i].g = rand() % 64;
		rand_banding[i].b = rand() % 255;
	}

	//STAGE 1) calculate the escape time for each pixel
#pragma omp parallel for private(i, x, c_r, c_i, n_r, n_i, o_r, o_i, mu) if(hist_method != OMP_SERIAL) schedule(dynamic, 1)
	for (y = 0; y < HEIGHT; y++)
	for (x = 0; x < WIDTH; x++)
	{
		//zero complex number values
		n_r = 0;
		n_i = 0;
		o_r = 0;
		o_i = 0;

		//calculate the initial real and imaginary part of z defined by the complex polynomial z-> z^2 + c  where c is the initial parameter based on zoom and displacement
		c_r = 1.5 * (x - WIDTH / 2) / (0.5 * ZOOM * WIDTH) + X_DISPLACEMENT;
		c_i = (y - HEIGHT / 2) / (0.5 * ZOOM * HEIGHT) + Y_DISPLACEMENT;

		//iterate to find how many iterations before outside the julia set
		for (i = 0; (i < MAX_ITERATIONS) && ((n_r * n_r + n_i * n_i) < ESCAPE_RADIUS_SQ); i++)
		{
			//store current values
			o_r = n_r;
			o_i = n_i;

			//apply mandelbrot function
			n_r = o_r * o_r - o_i * o_i + c_r;
			n_i = 2.0 * o_r * o_i + c_i;
		}

		//escape time algorithm if using HISTOGRAM_NORMALISED_ITERATION_COUNT transfer function or RANDOM_NORMALISED_ITERATION_COUNT
		if ((tf >= HISTOGRAM_NORMALISED_ITERATION_COUNT) && (i < MAX_ITERATIONS)) {
			mu = (double)i - log(log(sqrt(n_r*n_r + n_i*n_i))) / log(2);
			iterations_d[y][x] = mu;
			i = (int)mu;
		}

		iterations[y][x] = i;	//record the escape velocity

		if ((tf == HISTOGRAM_ESCAPE_VELOCITY) || (tf == HISTOGRAM_NORMALISED_ITERATION_COUNT)){
			switch (hist_method){
				//Exercise 2.2
			case(OMP_SERIAL) : {
								   histogram[i]++;
								   break;
			}
				//Exercise 2.2.1
			case(OMP_CRITICAL) : {
#pragma omp critical
									 {
										 histogram[i]++;
									 }
									 break;
			}
				//Exercise 2.2.2
			case(OMP_MASTER) : {
								   local_histogram[y][i]++;
								   if (i == 0){
									   printf("WTF\n");
								   }
								   break;
			}
				//Exercise 2.2.3
			case(OMP_ATOMIC) : {
#pragma omp atomic
								   histogram[i]++;
								   break;
			}
			}

		}

	}

	//Exercise 2.2.2 serial code for summing local histograms (performed by master only)
	if (hist_method == OMP_MASTER){
		for (y = 0; y < HEIGHT; y++)
		for (i = 0; i < MAX_ITERATIONS; i++)
			histogram[i] += local_histogram[y][i];

	}


	//STAGE 2) calculate the transfer (rgb output) for each pixel
#pragma omp parallel for private(x) schedule(dynamic)
	for (y = 0; y < HEIGHT; y++)
	for (x = 0; x < WIDTH; x++)
	{
		switch (tf){
		case (ESCAPE_VELOCITY) : {
									 rgb_output[y][x] = ev_transfer(x, y);
									 break;
		}
		case (HISTOGRAM_ESCAPE_VELOCITY) : {
											   rgb_output[y][x] = h_ev_tranfer(x, y);
											   break;
		}
		case (HISTOGRAM_NORMALISED_ITERATION_COUNT) : {
														  rgb_output[y][x] = h_nic_transfer(x, y);
														  break;
		}
		case (RANDOM_NORMALISED_ITERATION_COUNT) : {
													   rgb_output[y][x] = rand_nic_transfer(x, y);
													   break;
		}
		}
	}

	//STAGE 3) output the madlebrot to a file
	fwrite(rgb_output, sizeof(char), sizeof(rgb_output), f);
	fclose(f);

	//stop timer
	end = omp_get_wtime();

	elapsed = end - begin;
	printf("Complete in %f seconds\n", elapsed);

	return 0;
}


rgb ev_transfer(int x, int y){
	rgb a;
	double hue;
	int its;

	its = iterations[y][x];
	if (its == MAX_ITERATIONS){
		a.r = a.g = a.b = 0;
	}
	else{
		hue = its / (double)MAX_ITERATIONS;
		a.r = a.g = 0;
		a.b = (char)(hue * 255.0); //clamp to range of 0-255
	}
	return a;
}

rgb h_ev_tranfer(int x, int y){
	rgb a;
	double hue;
	int its;
	int i;

	its = iterations[y][x];
	if (its == MAX_ITERATIONS){
		a.r = a.g = a.b = 0;
	}
	else{
		hue = 0;
		for (i = 0; i < its; i++)
			hue += (histogram[i] / (double)(1024 * 768));
		a.r = a.g = 0;
		a.b = (char)(hue * 255.0); //clamp to range of 0-255
	}
	return a;
}

rgb h_nic_transfer(int x, int y){
	rgb a;
	double hue, hue1, hue2, its_d, frac;
	int i, its;

	its_d = iterations_d[y][x];
	its = iterations[y][x];

	hue1 = hue2 = 0;
	for (i = 0; (i < its) && (its<MAX_ITERATIONS); i++)
		hue1 += (histogram[i] / (double)(1024 * 768));
	if (i <= MAX_ITERATIONS)
		hue2 = hue1 + (histogram[i] / (double)(1024 * 768));
	a.r = a.g = 0;
	frac = its_d - (int)its_d;
	hue = (1 - frac)*hue1 + frac*hue2;	//linear interpolation between hues
	a.b = (char)(hue * 255.0);			//clamp to range of 0-255
	return a;
}

rgb rand_nic_transfer(int x, int y){
	rgb a;
	double r_hue, g_hue, b_hue, its_d;
	int its;

	its_d = iterations_d[y][x];
	its = iterations[y][x];

	r_hue = g_hue = b_hue = 0;
	if (its < MAX_ITERATIONS){
		double frac = its_d - (int)its_d;
		r_hue = (1 - frac)*(double)rand_banding[its].r + frac*(double)rand_banding[its + 1].r;
		g_hue = (1 - frac)*(double)rand_banding[its].g + frac*(double)rand_banding[its + 1].g;
		b_hue = (1 - frac)*(double)rand_banding[its].b + frac*(double)rand_banding[its + 1].b;
	}
	a.r = (char)(r_hue);
	a.g = (char)(g_hue);
	a.b = (char)(b_hue);
	return a;
}
