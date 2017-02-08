#include <stdio.h>


#include "random.h"

#define NUM_VALUES 250
int values [NUM_VALUES];

int main()
{

	unsigned int sum = 0;
	unsigned int average = 0;
	int min = 0;
	int max = 0;
	unsigned char i = 0;	

	init_random();

	for (i=0; i<NUM_VALUES; i++){
		values[i] = random_ushort();
		
		sum += values[i];
	}

	average = sum / NUM_VALUES;

	for (i=0; i<NUM_VALUES; i++){
		values[i] -= average;
		min = (values[i] < min)? values[i] : min;
		max = (values[i] > max)? values[i]: max;
	}

	printf ("Sum is %u\n", sum);
	printf ("Average is %u\n", average);
	printf ("Min is %d\n", min);
	printf ("Max is %d\n", max);
    return 0;
}
