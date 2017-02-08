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
	f = fopen("commands.calc", "r");
	if (f == NULL){
		fprintf(stderr, "File not found\n");
		return;
	}


    while (readLine(f, buffer)){
		line++;

		if (!(isalpha(buffer[0]) &&  isalpha(buffer[1]) && isalpha(buffer[2]) && buffer[3]==' ')){
			fprintf(stderr, "Incorrect command format at line %u\n", line);
			return;
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
		
		else{
			fprintf(stderr, "Unknown command at line %u!\n", line);
			return;
		}

	}

	printf("Final sum is %f\n", sum);
	
	fclose(f);

    return 0;
}

int readLine(FILE *f, char buffer[]){
	int i=0;
	char c;
	while ((c = getc(f)) != '\n'){
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