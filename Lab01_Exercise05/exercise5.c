#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define BUFFER_SIZE 32

int readLine(FILE *f, char buffer[]);

int main()
{
	FILE *f;
    float in_value, sum;
	unsigned int line;
	char buffer [BUFFER_SIZE];
	char command [4];
    
	sum = 0;
	line = 0;

	printf("Welcome to basic COM4521 calculator\nEnter command: ");
	// Ex 5.1.1 (1/2) fopen() is used to open a file, returning a file handle
	// The mode for reading is specified using "r"
	// See documentation: https://www.cplusplus.com/reference/cstdio/fopen/
	f = fopen("commands.calc", "r");
	if (f == NULL){
		fprintf(stderr, "File not found\n");
		return EXIT_FAILURE;
	}


    while (readLine(f, buffer)){
		line++;

		if (!(isalpha(buffer[0]) &&  isalpha(buffer[1]) && isalpha(buffer[2]) && buffer[3]==' ')){
			fprintf(stderr, "Incorrect command format at line %u\n", line);
			return EXIT_FAILURE;
		}

		sscanf(buffer, "%s %f", command, &in_value);

		if (strcmp(command, "add")==0){
			sum += in_value;
		}else if (strcmp(command, "sub")==0){
			sum -= in_value;
		}else if (strcmp(command, "div")==0){
			sum /= in_value;
		}else if (strcmp(command, "mul")==0){
			sum *= in_value;
		}
		// Ex 5.1.3, If the command is not understood, report an error and close the program returning EXIT_FAILURE
		// EXIT_SUCCESS and EXIT_FAILURE are the core returns codes in C, however any non-zero integer is considered an exit failure
		else{
			fprintf(stderr, "Unknown command at line %u!\n", line);
			return EXIT_FAILURE;
		}

	}

	printf("Final sum is %f\n", sum);

	// Ex 5.1.1 (22) fclose() is used to close an open file, by passing the file handle that was provided by fopen()
	// See documentation: https://www.cplusplus.com/reference/cstdio/fclose/
	fclose(f);

    return EXIT_SUCCESS;
}

int readLine(FILE *f, char buffer[]){
	int i=0;
	char c;
	while ((c = getc(f)) != '\n'){
		// Ex 5.1.2, Files have their end marked with EOF, rather than the null terminating character (`\0`)
		// In this particular case, we expect the end of file after `\n`, so the file must end with a new line
		// A more complicated update to the function could be applied, to handle EOF anywhere in the file
		if (c == EOF)
			return 0;
        buffer[i++] = c;
		if (i == BUFFER_SIZE){
			fprintf(stderr, "Buffer size is too small for line input\n");
			exit(0);
		}
	}
	buffer[i] = '\0';

	if (strncmp(buffer, "exit", 4)==0)
		return 0;
	else
		return 1;

}