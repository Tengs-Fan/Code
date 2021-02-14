#include <unistd.h>
#include <sys/types.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

int
main( int argc , char* argv[])
{
	fprintf(stderr, "EACCES: %s \n", strerror(EACCES));

	char s[512] = "ahhah";
	errno = ENOENT;
	//perror(s);
	perror(argv[1]);

	return 0;
}
