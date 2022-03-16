#include "threads.h"
#include "common_threads.h"
#include "semaphore.h"
#include "stdio.h"
#include "unistd.h"

sem_t s;

void *child(void *arg) {
	printf("child\n");
	sleep(1);
	sem_post(&s);
	/*sleep(1);*/
	printf("child, after post\n");
	return NULL;
}

int main() {
	sem_init(&s, 0, 1);
	printf("Parent: begin\n");
	pthread_t c;
	Pthread_create(&c, NULL, child, NULL);
	/*sleep(1);*/

	sem_wait(&s);
	printf("Parent: end\n");
}
