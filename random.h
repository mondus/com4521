#include <stdlib.h>

#define RAND_SEED 123

void init_random(){
	srand(RAND_SEED);
}

unsigned short random_ushort(){
	return rand();
}


