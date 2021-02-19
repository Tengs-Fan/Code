#include <unistd.h>
#include <stdio.h>
#define exit(EXP)	return(EXP)
#define err_sys(EXP) { printf(EXP); \
	return 0;}
