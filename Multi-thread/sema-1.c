#include "threads.h"
#include "common_threads.h"
#include "semaphore.h"
#include "stdio.h"

sem_t s;

void *child(void *arg) {
	printf("child\n");
	sem_post(&s);
	return NULL;
}

int main() {
	sem_init(&s, 0, 1);
	printf("Parent: begin\n");
	pthread_t c;
	Pthread_create(&c, NULL, child, NULL);

	sem_wait(&s);
	printf("Parent: end\n");
}
