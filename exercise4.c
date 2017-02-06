#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

		//4.5 Check that the line contains 3 letters and a spaceextract
		//4.6 Extract the command and in_value using sscanf
		if (false){ //4.7 Change condition to check command to see if it is "add"
			sum += in_value;
		}
		//4.8 Add else if conditions for sub, mul and div
		}else{
			printf("Unknown command\n");
		}

		printf("\tSum is %f\n", sum);
		printf("Enter next command: ");
	}

    return 0;
}

int readLine(char buffer[]){
	int i=0;
	char c=0;
	while ((c = getchar()) != '\n'){
        //4.1 Add character to buffer
		//4.2 Check index for overflow
	}
	//4.3 Ensure the buffer is correctly terminated

	//4.4 Return 0 if buffer = "exit" otherwise return 1

}