// Ex 2.3, As noted in the previous exercise, pre-processor macros are created in the form `#define <name> <replacement>`
#define RAND_SEED 123
#define RANDOM_A 1103515245 
#define RANDOM_C 12345 

// Ex 2.2, These two function prototypes, declare to the compiler that a function with matching name, arguments and return type is defined elsewhere
// Knowing this, the compiler will accept calls to the function, assuming they can be resolved during the linker stage.
// This allows a function to be called in a separate module (.c file), than the one which it is defined
// The extern keyword, notifies the compiler that something is externally defined, this is implied by a function prototype so not required.
extern void init_random();
extern unsigned short random_ushort();
// Ex 2.5 (1/2), A further prototype is added for our new random function
extern unsigned int random_uint();


