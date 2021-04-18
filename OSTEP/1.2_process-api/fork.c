#include <unistd.h>
#include <stdio.h>

int 
main(){
	int change = 0x55aa;
	pid_t pid;

	int rc = fork();
	if (rc < 0){
		fprintf(stderr, "fork failed!\n");
	} else if (rc == 0) {
		change = 0xABAB;
		fprintf(stdout, "Child, Variable: 0x%x\n",change);
	} else {
//		change = 0xCDCD;
		fprintf(stdout, "Parent, Variable: 0x%x\n",change);
	}

}
