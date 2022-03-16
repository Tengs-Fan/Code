#include "common_threads.h"
#include "stdio.h"

typedef struct __counter_t {
	int 			value;
}counter_t;

void increment(counter_t *c) {
	c->value ++;
}

counter_t counter;

int addNumber() {
	for (int i=0; i<40000000; i++) {
		increment(&counter);
	}
	return counter.value;
}

void decNumber() {
	for (int i=0; i<20000000; i++) {
		increment(&counter);
	}
}

int main() {
	addNumber();
	int ret = counter.value;
	printf("return %d\n", ret);
	return 0;
}

