#include "stdio.h"


int main(){
	unsigned long lo, hi;
	double cpu_time = 0.0;
	__asm__ __volatile__("rdtsc":"=a"(lo),"=d"(hi));
	cpu_time = (double) (lo| (hi << 32));
	__asm__ __volatile__("nop");
	__asm__ __volatile__("rdtsc":"=a"(lo),"=d"(hi));
	cpu_time -= (double) (lo| (hi << 32));

	printf("time: %f",cpu_time);
}
