#include <stdio.h>
#include <stdlib.h>
#include "linked_list.h"

struct student{
	char *forename;
	char *surname;
	float average_module_mark;
};

void print_student(const struct student *s);

// Ex 3.3, students2.bin is opened with fopen()
// It is then iterated over using fread(), see the individual comments to understand how
void main(){
	llitem *end;
	llitem *first;
	llitem *ll;
	unsigned int str_len;

	end = NULL;
	first = NULL;

	//set callback function for printing using explicit cast
	print_callback = (void (*)(void *))&print_student;

	FILE *f = NULL;
	f = fopen("students2.bin", "rb"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find students2.bin file \n");
		exit(1);
	}

	//read student data
	// fread first reads the length of the forename (and stores it in the str_len variable) else there are no more records and we will break from the loop
	while (fread(&str_len, sizeof(unsigned int), 1, f) == 1){
		//allocate a new student structure to hold the record from file
		struct student *s;
		s = (struct student *)malloc(sizeof(struct student));

		//allocate enough space in the pointer to char forename to hold the string data from file
		s->forename = (char*)malloc(sizeof(char)*str_len);
		//read str_len characters from the file and store in the memory pointed to by forename
		fread(s->forename, sizeof(char), str_len, f);
		
		//read the length of the surname (and stores it in the str_len variable)
		fread(&str_len, sizeof(unsigned int), 1, f);
		//allocate enough space in the pointer to char surname to hold the string data from file
		s->surname = (char*)malloc(sizeof(char)*str_len);
		//read str_len characters from the file and store in the memory pointed to by surname
		fread(s->surname, sizeof(char), str_len, f);

		//read the module mark (and stores it in the student structure average_module_mark variable)
		fread(&s->average_module_mark, sizeof(float), 1, f);

		//append a new item to the linked list
		if (end == NULL){
			end = create_linked_list();
			first = end;
		}
		else
			end = add_to_linked_list(end);

		//set the record (cast the pointer to the student structure as a generic void pointer)
		end->record = (void *)s;

	}
	fclose(f);

	//print
	print_items(first);

	//cleanup records
	ll = first;
	while (ll != NULL){
		free(((struct student*)ll->record)->forename);
		free(((struct student*)ll->record)->surname);
		free((struct student*)ll->record);
		ll = ll->next;
	}
	free_linked_list(first);

}

void print_student(const struct student *s){
	printf("Student:\n");
	printf("\tForename: %s\n", s->forename);
	printf("\tSurname: %s\n", s->surname);
	printf("\tAverage Module Mark: %.2f:\n", s->average_module_mark);
}

