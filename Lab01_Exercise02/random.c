// Ex 2.1 (1/3), These includes are required by the two functions that have been copied to this file so they must be copied too
// (random.h is only required as of Ex 2.3, for the pre-processor macros)
#include <stdlib.h>
#include "random.h"

// Ex 2.4 (1/2), as this variable is defined outside of a method, it is at global scope, and accessible by code following it in the module (.c file)
// If `extern unsigned int rseed;` were used in a different module, it could be accessed there too.
unsigned int rseed;

// Ex 2.1 (2/3), This function has been copied directly from Exercise 1's random.h
// It is important that it's prototype `void init_random()` matches that placed in random.h
// So that the compiler can resolve function at the linker stage
void init_random(){
	srand(RAND_SEED);
	// Ex 2.4 (2/2), this sets the value of rseed
	// rseed could be initialised to this value where it declared above, however in this case that appears unnecessary,
    // as a user may call init_random() multiple times to reset the random stream, so this would still be required.
	rseed = RAND_SEED;
}

// Ex 2.1 (3/3), This function has also been copied directly from Exercise 1's random.h
unsigned short random_ushort(){
	return (unsigned short) (rand());
}

// Ex 2.5 (2/2), A function is implemented, which has a prototype matching that added to random.h
// This function implements a linear congruential generator, a simple means of pseudo-random stream generation
unsigned int random_uint(){
	rseed = RANDOM_A*rseed + RANDOM_C;
	return rseed;
}
