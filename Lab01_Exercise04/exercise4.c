#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define BUFFER_SIZE 32

int readLine(char buffer[]);

int main()
{
    float in_value, sum;
	char buffer [BUFFER_SIZE];
	char command [4];
    sum = 0;

	printf("Welcome to basic COM4521 calculator\nEnter command: ");

    while (readLine(buffer)){

		// Ex 4.5, Check the value of the first 4 elements of the buffer
		// isalpha() is a utility function found in the system header ctype.h, which returns true if the passed character is alphabetic
		// See documentation: https://www.cplusplus.com/reference/cctype/isalpha/
		if (!(isalpha(buffer[0]) &&  isalpha(buffer[1]) && isalpha(buffer[2]) && buffer[3]==' ')){
			fprintf(stderr, "Incorrect command format\n");
			continue;
		}

		// Ex 4.6, sscanf() reads in data according to a specified format
		// The format strings use the same format specifiers as printf()
		// The return values are provided to the trailing arguments, the pointer to these variables must be passed
		// So `&in_value` is used to get the address in memory where the variable in_value resides.
		// This allows sscanf() to write to the in_value
		// See documentation: https://www.cplusplus.com/reference/cstdio/sscanf/
		sscanf(buffer, "%s %f", command, &in_value);
		// Ex 4.7 & 4.8, strcmp() is used to provide more comparisons
		// Note, here strcmp() is used, rather than strncmp()
		// The difference is that strncmp() requires a length to be specified
		// In contrast, strncmp() looks for the null terminating character to assume length
	    // See documentation: https://www.cplusplus.com/reference/cstring/strcmp/
		if (strcmp(command, "add")==0){
			sum += in_value;
		}else if (strcmp(command, "sub")==0){
			sum -= in_value;
		}else if (strcmp(command, "div")==0){
			sum /= in_value;
		}else if (strcmp(command, "mul")==0){
			sum *= in_value;
		}
		// Ex 4.9, Further comparisons using strncmp()
		else if (strncmp(command, "ad", 2)==0){
			printf("Did you mean add?\n");
			continue;
		}else if (strncmp(command, "su", 2)==0){
			printf("Did you mean sub?\n");
			continue;
		}else if (strncmp(command, "mu", 2)==0){
			printf("Did you mean mul?\n");
			continue;
		}else if (strncmp(command, "di", 2)==0){
			printf("Did you mean div?\n");
			continue;
		}
		
		else{
			fprintf(stderr, "Unknown command!\n");
			continue;
		}

		printf("\tSum is %f\n", sum);
		printf("Enter command: ");
	}

    return 0;
}

int readLine(char buffer[]){
	int i=0;
	char c;
	while ((c = getchar()) != '\n'){
		// Ex 4.1, Copy the character c to location i in buffer
		// C allows you to to increment using `i++` or `++i`, it is important to understand the difference
		// `i++` makes a copy of i, increments i, then returns the copy of i (it's value pre increment)
		// `++i` increments i and returns it's new value (it's value post increment)
		// As `++i` does not produce a copy, it is slightly more efficient, so often seen used in for loops where the return value is not required
        buffer[i++] = c;
		// Ex 4.2, The buffer has length BUFFER_SIZE, so if i >= BUFFER_SIZE, the write will be beyond the end of the array
		// C, unlike higher level programming languages, will not normally throw an exception if you read/write out of memory bounds
		// This can lead to unrelated variables being corrupted, or even OS level security vulnerabilities (see 'buffer overflow' on wikipedia)
		// Therefore it is always important to consider the arithmetic used to access arrays, to ensure accesses will always be within bounds.
		if (i == BUFFER_SIZE){
			fprintf(stderr, "Buffer size is too small for line input\n");
			exit(0);
		}
	}
	// Ex 4.3, Single quotes are used to provide a character literal.
	// The character literal `\0`is known as the null terminating character
	// This evaluates to 0, and is recognised by most C functions which take strings to denote the end of the string, to prevent reading out of bounds.
	// The check added in exercise 4.2, ensure that i will be in bounds at this point, so no additional check is required
    buffer[i] = '\0';

	// Ex 4.4, strncmp() returns an integer, denoting the comparison of two strings
	// If the strings match, 0 is returned, otherwise a positive/negative number denotes an order
	// See documentation: https://www.cplusplus.com/reference/cstring/strncmp/
	// When string literals are defined like "exit", they implictly include the null terminating character `\0`
    // So if considering the length, "exit" would have a length of 5
	if (strncmp(buffer, "exit", 4)==0)
		return 0;
	else
		return 1;

}