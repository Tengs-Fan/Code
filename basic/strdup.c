#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(){
	char *s;

	s = strdup("Hallo Welt");
	if(s == NULL){
		fprintf(stderr,"Can't allocate memory with malloc\n.");
		return (EXIT_FAILURE);
	}
//	printf("%p\n",(void *)s);
//	return (EXIT_SUCCESS);

	unsigned long int i=0;
	while(s){
		printf("[%lu] %s (%p)\n",i,s,(void *)s);
		sleep(1);
		i++;
	}
	return (EXIT_SUCCESS);
}
