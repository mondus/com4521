#include <stdio.h>
#include <stdlib.h>


typedef struct llitem{
	struct llitem *previous;
	struct llitem *next;
	void* record;
} llitem;

// Exercise 1.2) a function pointer which accepts a generic pointer and returns void
void (*print_callback)(void* r) = NULL;

void print_items(llitem *ll_start){
	llitem *ll = ll_start;
	while (ll != NULL){
		//printf
		if (print_callback != NULL)
			print_callback(ll->record);
		//next
		ll = ll->next;
	}
}

llitem* create_linked_list(){
	llitem* ll_start;

	ll_start = (llitem*)malloc(sizeof(llitem));
	ll_start->next = NULL;
	ll_start->previous = NULL;
	ll_start->record = NULL;
	return ll_start;
}

/* Exercise 1.1 */
llitem* add_to_linked_list(llitem* ll_end){
	llitem *ll;
	// pre check to make sure that the pointer provided points to a llitem struct which has been allocated
	if (ll_end == NULL)
		return NULL;
	// Exercise 1.1.1) Check that the ll_end item is in fact the end of the list (the next record should be NULL). 
	if (ll_end->next != NULL)
		// If it is not the end then the function should return NULL. 
		return NULL;
	// Allocate a new llitem (the new end of the list)
	ll = (llitem*)malloc(sizeof(llitem));
	// updating the old end of the linked list 
	ll_end->next = ll;
	// update the new end of the list to pint backwards to the old end
	ll->previous = ll_end;
	//set next pointer of the new end of the linked list to point to NULL
	ll->next = NULL;
	//set the record to NULL
	ll->record = NULL;
	return ll;
}

void free_linked_list(llitem *ll_start){
	llitem *ll = ll_start;
	while (ll != NULL){
		llitem *temp = ll->next;
		free(ll);
		ll = temp;
	}
}

