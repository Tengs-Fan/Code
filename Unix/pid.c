#include <unistd.h>
#include <sys/types.h>
#include <stdio.h>

int
main( int argc , char* argv[])
{
	printf("Hello from process %ld\n",(long)getpid());

	return 0;
}
