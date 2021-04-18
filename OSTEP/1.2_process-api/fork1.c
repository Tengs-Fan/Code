#include <unistd.h>
#include <sys/wait.h>
#include <stdio.h>

int 
main(){
	int change = 0x55aa;
	pid_t pid = getpid();

	int rc = fork();
	if (rc < 0){
		fprintf(stderr, "fork failed!\n");
	} else if (rc == 0) {
		change = 0xABAB;
		//wait(NULL);
		waitpid(pid, NULL, NULL);
		fprintf(stdout, "Child, Variable: 0x%x\n",change);
	} else {
		change = 0xCDCD;
		sleep(1);
		fprintf(stdout, "Parent, Variable: 0x%x\n",change);
	}

}
