#include "common_threads.h"
#include "stdio.h"
#include "stdlib.h"
#include "sys/time.h"

#define THREAD_POOL_SIZE 128

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
	int threadNum = *(int*)arg;
	int threadID = ((int*)arg)[1];
	for (int i=0; i<12800000 / threadNum; i++) {
		increment(&counter);
	}
	printf("I am here: %d\n", threadID);
}

int main(int argc, char** argv) {

	struct timeval begin, end;
	double cpu_time = 0.0;
	int ret = gettimeofday(&begin, NULL);

	pthread_t p[THREAD_POOL_SIZE];
	//int* args = malloc(sizeof(int) * 2 * (THREAD_POOL_SIZE) );
	int args[2*THREAD_POOL_SIZE];
	int threadNum = atoi(argv[1]);
	init(&counter);

	for (int i = 0; i < threadNum; i++)
	{
		args[2*i] = threadNum;
		args[2*i+1] = i;
		Pthread_create(&p[i], NULL, (void*)addNumber, (void*) (&args[2*i]));
	}

	for (int i = 0; i < threadNum; i++)
	{
		Pthread_join(p[i], NULL);
	}
	ret = get(&counter);

	gettimeofday(&end, NULL);
	cpu_time = (end.tv_sec - begin.tv_sec) * 1000;
	cpu_time += (end.tv_usec - begin.tv_usec) / 1000;
	printf("Return %d\n Using CPU in %lf ms \n", ret, cpu_time);
	return 0;
}
