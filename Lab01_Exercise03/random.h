#define RAND_SEED 123
#define RANDOM_A 1103515245 
#define RANDOM_C 12345 

extern void init_random();

extern unsigned short random_ushort();

extern unsigned int random_uint();

// Ex 3.1 (1/3), A further prototype has been added to the header
extern float random_float();


