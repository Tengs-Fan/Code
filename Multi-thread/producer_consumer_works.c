#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>

#include "threads.h"
#include "common_threads.h"

#include <semaphore.h>

#define MAX 64

int buffer[MAX];
int fill = 0;
int use = 0;

void put(int value) {
	buffer[fill] = value;
	fill = (fill + 1) % MAX;
}

int get() {
	int tmp = buffer[use];
	use = (use + 1) % MAX;
	return tmp;
}

sem_t empty;
sem_t full;

void *producer(void *arg) {
	for (int i = 0; i < )
}
