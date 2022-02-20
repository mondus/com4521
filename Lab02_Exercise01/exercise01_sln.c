#include <stdio.h>
#include <stdlib.h>

#define NUM_STUDENTS 4

struct student{
	char forename[128];
	char surname[128];
	float average_module_mark;

};

void print_student(const struct student *s);

void main(){
	// Ex 1.3, (1/2) To declare a pointer, as opposed to an object, * is inserted between the type and identifier
	// C requires a struct's type to be specified in the form `struct <identifier>`, this can be avoided with typedef
	struct student *students;
	int i;

	// Ex 1.3, (2/2) In order for a pointer to be valid, it must point to valid memory
	// This can either be achieved by referencing an object (using &, see Ex 1.2 (1/2))
	// or you can manually allocate memory by calling malloc() and specifying the number of bytes required
	students = (struct student*)malloc(sizeof(struct student)*NUM_STUDENTS);

	FILE *f = NULL;
	f = fopen("students.bin", "rb"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find students.bin file \n");
		exit(1);
	}

	fread(students, sizeof(struct student), NUM_STUDENTS, f);
	fclose(f);

	for (i = 0; i < NUM_STUDENTS; i++){
		// Ex 1.2, (1/2) To get the pointer to students[i], the address it resides in memory, it must be referenced using &
		print_student(&students[i]);
	}

	free(students);
}

// Ex 1.2, (2/2)  To declare a pointer, as opposed to an object, * is inserted between the type and identifier
// A pointer is a reference to a location in memory, and can be manipulated with arithmetic operators
// e.g. to access the memory like an array
// C++ introduces references, which are distinct from pointers, so despite use of the reference operator (Ex 1.2 (1/2)),
// normally they will be called pointers.
void print_student(const struct student *s){
	printf("Student:\n");
	printf("\tForename: %s\n", s->forename);
	printf("\tSurname: %s\n", s->surname);
	printf("\tAverage Module Mark: %.2f:\n", s->average_module_mark);
}

