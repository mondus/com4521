#include <stdlib.h>

#define RAND_SEED 123

void init_random(){
    srand(RAND_SEED);
}

unsigned short random_ushort(){
// Ex 1.7, C-style casts take the form `(<type>)`, this is the same as Java
    return (unsigned short) (rand());
}


