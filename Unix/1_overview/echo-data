#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUFF_SIZE 4096

int
main(void)
{
	int n;
	char buf[BUFF_SIZE];

	while (( n = fread(STDIN_FILENO,BUFF_SIZE,buf,BUFF_SIZE)) > 0){
		if ( fwrite(STDOUT_FILENO,BUFF_SIZE, buf, n) != n){
			printf("write error");
			exit(1);
		}
	}
	if (n<0){
		printf("read error");
		exit(2);
	}

	return 0;
}
