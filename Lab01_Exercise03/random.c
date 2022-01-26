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
// Ex 3.1 (2/3), The function simply casts the result from random_uint() to float
// Therefore the random number will still be a whole number,
// this is  opposed to the norm for random floats which are usually returned in the range [0, 1)
float random_float(){
	return (float) random_uint();
}
