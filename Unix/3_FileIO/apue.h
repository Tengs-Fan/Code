#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#define exit(EXP)	return(EXP)
#define err_sys(EXP) { printf(EXP); \
	return 0;}
#define err_quit(EXP) { printf(EXP); \
	return 0;}
#define err_dump(EXP) { printf(EXP); \
	break ;}
