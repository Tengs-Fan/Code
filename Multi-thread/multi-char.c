#include <stdio.h>

#include "common_threads.h"

void thread1() { while (1) { printf("a"); } }
void thread2() { while (1) { printf("b"); } }

int main(int argc, char *argv[]) {
    pthread_t p1, p2;
    Pthread_create(&p1, NULL, thread1, (void *) (long long) 0);
    Pthread_create(&p2, NULL, thread2, (void *) (long long) 1);
    Pthread_join(p1, NULL);
    Pthread_join(p2, NULL);
    return 0;
}
