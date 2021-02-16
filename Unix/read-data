#include <stdio.h>

int
main(void){
	int c;

	while((c = getc(stdin)) != EOF){
		if (putc(c,stdout) == EOF){
			printf("Output Error");
		}
	}

	if (ferror(stdin))
	{
		printf("Input Error");
	}

	return 0;
}
