#include "common_threads.h"
#include "stdio.h"
#include "stdlib.h"

typedef struct __counter_t {
	int 			value;
	pthread_mutex_t lock;
}counter_t;

void init (counter_t *c) {
	c->value = 0;
	Pthread_mutex_init(&c->lock, NULL);
}

void increment (counter_t *c) {
	Pthread_mutex_lock (&c->lock);
	c->value++;
	Pthread_mutex_unlock (&c->lock);
}

void decrement (counter_t *c) {
	Pthread_mutex_lock (&c->lock);
	c->value -- ;
	Pthread_mutex_unlock (&c->lock) ;
}

int get (counter_t *c) {
	Pthread_mutex_lock (&c->lock) ;
	int rc = c->value;
	Pthread_mutex_unlock (&c->lock) ;
	return rc;
}

counter_t counter;

void addNumber(void* arg) {
	int ID = *((int*)arg);
	for (int i=0; i<12800000 / *((int*)arg+1); i++) {
		increment(&counter);
	}
	printf("I am here: %d\n", ID);
}

void decNumber() {
	for (int i=0; i<20000000; i++) {
		increment(&counter);
	}
}

int main(int argc, char** argv) {
	pthread_t p[128];
	init(&counter);

	int* args = malloc(sizeof(int) * 2);
	args[1] = (int)atoi(argv[1]);
	for (int i = 0; i < atoi(argv[1]); i++)
	{
		args[0] = i;
		Pthread_create(&p[i], NULL, (void*)addNumber, (void*) args);
	}
	for (int i = 0; i < atoi(argv[1]); i++)
	{
		Pthread_join(p[i], NULL);
	}
	int ret = get(&counter);
	printf("return %d\n", ret);
	return 0;
}
