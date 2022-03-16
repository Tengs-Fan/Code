#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "common_threads.h"
#include "zemaphore.h"

// If done correctly, each child should print their "before" message
// before either prints their "after" message. Test by adding sleep(1)
// calls in various locations.

// You likely need two semaphores to do this correctly, and some
// other integers to track things.

typedef struct __barrier_t {
	Zem_t sema;
	Zem_t all;
	int thread_num;
	int all_arrived;
    // add semaphores and other information here
} barrier_t;


// the single barrier we are using for this program
barrier_t b;

void barrier_init(barrier_t *b, int num_threads) {
	b->thread_num = num_threads;
	b->all_arrived = 0;
	Zem_init(&b->sema, num_threads);
	Zem_init(&b->all, 0);
    // initialization code goes here
}

void barrier(barrier_t *b) {
	Zem_wait(&b->sema);
	b->thread_num --;
	if (b->thread_num <= 0) {
		
	}
    // barrier code goes here
}

//
// XXX: don't change below here (just run it!)
//
struct tinfo_t {
    int thread_id;
} ;

void *child(void *arg) {
    struct tinfo_t *t = (struct tinfo_t *) arg;
    printf("child %d: before\n", t->thread_id);
    barrier(&b);
    printf("child %d: after\n", t->thread_id);
    return NULL;
}


// run with a single argument indicating the number of
// threads you wish to create (1 or more)
int main(int argc, char *argv[]) {
    assert(argc == 2);
    int num_threads = atoi(argv[1]);
    assert(num_threads > 0);

    pthread_t p[num_threads];
    struct tinfo_t t[num_threads];

    printf("parent: begin\n");
    barrier_init(&b, num_threads);

    int i;
    for (i = 0; i < num_threads; i++) {
	t[i].thread_id = i;
	Pthread_create(&p[i], NULL, child, &t[i]);
    }

    for (i = 0; i < num_threads; i++)
	Pthread_join(p[i], NULL);

    printf("parent: end\n");
    return 0;
} 
