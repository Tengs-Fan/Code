#ifdef __GNUC__
#include<alloca.h>
#endif

#include<stdio.h>

int main() {
	
	char *buf = (char*)alloca(600);
	snprintf(buf,600, "hi, %d, %d, %d", 1,2,3);

	puts(buf);
	}
