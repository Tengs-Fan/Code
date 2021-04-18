#include <unistd.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <stdio.h>

int 
main(){
	int change = 0x55aa;
	pid_t pid = getpid();
	int pipefd[2];
	char buf[1024];

	if (pipe[pipefd] == -1) {
		perror("Pipe Failed!\n");
		exit(EXIT_FAILURE);
	}

	{
		int rc = fork();
		if (rc < 0){
			fprintf(stderr, "fork failed!\n");
		} else if (rc == 0) {
			close(pipefd[1]);
			
			while(read(pipefd[0], buf, sizeof(int)) > 0)
				write(STDOUT_FILENO, buf, 1);
			change = 0x0101;
			fprintf(stdout, "Child, Variable: 0x%x\n",change);
		} else {
			change = 0xCDCD;
			sleep(1);
			fprintf(stdout, "Parent, Variable: 0x%x\n",change);
		}
	}
	{
		int rc = fork();
		if (rc < 0){
			fprintf(stderr, "fork failed!\n");
		} else if (rc == 0) {
			change = 0x0202;
			fclose(STDOUT_FILENO);
			fprintf(stdout, "Child2, Variable: 0x%x\n",change);
		} else {
			change = 0xCCCC;
			sleep(1);
			fprintf(stdout, "Parent2, Variable: 0x%x\n",change);
		}
	}
}
