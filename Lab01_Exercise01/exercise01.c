#include <stdio.h>

// Ex 1.5 `#include` statements, cause the specified file to be recursively copied into the source file at the location of the statement prior to compilation
// For this reason it's normal for headers to use macros to prevent them from being included multiple times in the header hierarchy.
#include "random.h"

// Ex 1.1, Pre-processor definitions (or Macros) take the form `#define <name>` where the remainder of the line provides the replacement
// The compiler's pre-processor, will replace occurrences of `NUM_VALUES` with `250` in the source prior to compilation
#define NUM_VALUES 250
// Ex 1.2, Arrays declared in this form must have a known size at compile time
// In this case, the compiler will know the length to be 250, due to the replacement of `NUM_VALUES`
int values [NUM_VALUES];

int main()
{
    // Ex 1.3, sum is declared, local to the function main(), and initialised to 0
	// As an unsigned integer, it cannot hold negative values, but has a greater positive range than a signed integer
	unsigned int sum = 0;
	// Ex 1.8 (2/3), an unsigned int has [0, 2^32) range, it should be suitable to hold the sum of 250 unsigned shorts
	// Early C specifications required all variables to be declared at the start, this is nolonger strictly enforced by most compilers
	unsigned int average;
	int min = 0;
	int max = 0;
	// Ex 1.4, a char is 8 bits, compared to an int which is 32 bits, therefore i can represent the inclusive range [0, 255]
	// `i` is used as the looping variable, normally this would be declared in the form `for(int i =0;`,
    // however this is not supported according to early C specifications, so may not be permitted by some compilers.
	unsigned char i = 0;	

	init_random();

	// Ex 1.6, This is a basic C-style for loop
	for (i=0; i<NUM_VALUES; i++){
		values[i] = random_ushort();
		// Check the documentation to get the right format specifiers for unusual types
		// https://www.cplusplus.com/reference/cstdio/printf/
		// printf("%hhu: %hu\n", i, values[i]);

		// Ex 1.8 (1/3), Add the value stored in values[i] to sum
		// Functionally equivalent shorthand for `sum = sum + values[i];`
		sum += values[i];
	}

	// Ex 1.8 (3/3)
	average = sum / NUM_VALUES;

	// Ex 1.9
	for (i=0; i<NUM_VALUES; i++){
		values[i] -= average;
		// Ternary statements (inline if statements) take the form `(conditional expression)? expression_t : expression_f ;`
		min = (values[i] < min)? values[i] : min;
		max = (values[i] > max)? values[i] : max;
	}

	// Unsigned int requires the format specifier %u, and int requires %d
	printf("Sum is %u\n", sum);
	printf("Average is %u\n", average);
	printf("Min is %d\n", min);
	printf("Max is %d\n", max);
    return 0;
}
