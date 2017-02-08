#include <stdlib.h>
#include "random.h"

unsigned int rseed;

void init_random(){
	srand(RAND_SEED);
	rseed = RAND_SEED;
}

unsigned short random_ushort(){
	return (unsigned short) (rand());
}

unsigned int random_uint(){
	rseed = RANDOM_A*rseed + RANDOM_C;
	return rseed;
}

float random_float(){
	return (float) random_uint();
}
