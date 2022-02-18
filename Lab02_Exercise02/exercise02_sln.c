#include <stdio.h>
#include <stdlib.h>

#define NUM_STUDENTS 4

struct student{
	// Ex 2, (1/4) forename and surname are now pointers, these will need to be initialised
	char *forename;
	char *surname;
	float average_module_mark;

};

void print_student(const struct student *s);

void main(){
	struct student *students;
	int i;

	students = (struct student*)malloc(sizeof(struct student)*NUM_STUDENTS);

	FILE *f = NULL;
	f = fopen("students2.bin", "rb"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find students2.bin file \n");
		exit(1);
	}

	//read student data
	for (i = 0; i < NUM_STUDENTS; i++){
		unsigned int str_len;
		fread(&str_len, sizeof(unsigned int), 1, f);
		// Ex 2, (2/4) forename is now initialised using malloc() to the desired length
		students[i].forename = (char*)malloc(sizeof(char)*str_len);
		fread(students[i].forename, sizeof(char), str_len, f);
		fread(&str_len, sizeof(unsigned int), 1, f);
		// Ex 2, (3/4) surname is now initialised using malloc() to the desired length
		students[i].surname = (char*)malloc(sizeof(char)*str_len);
		fread(students[i].surname, sizeof(char), str_len, f);
		fread(&students[i].average_module_mark, sizeof(float), 1, f);

	}
	fclose(f);

	//print
	for (i = 0; i < NUM_STUDENTS; i++){
		print_student(&students[i]);
	}

	//cleanup
	for (i = 0; i < NUM_STUDENTS; i++){
		// Ex 2, (4/4) Manually allocated memory, must also be manually deallocated, failing to do so can create memory leaks.
		free(students[i].forename);
		free(students[i].surname);
	}
	free(students);

}

void print_student(const struct student *s){
	printf("Student:\n");
	printf("\tForename: %s\n", s->forename);
	printf("\tSurname: %s\n", s->surname);
	printf("\tAverage Module Mark: %.2f:\n", s->average_module_mark);
}

