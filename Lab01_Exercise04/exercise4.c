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

		if (!(isalpha(buffer[0]) &&  isalpha(buffer[1]) && isalpha(buffer[2]) && buffer[3]==' ')){
			fprintf(stderr, "Incorrect command format\n");
			continue;
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