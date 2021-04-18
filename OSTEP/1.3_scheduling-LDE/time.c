#include <unistd.h>

int main(){
	char buf[32];
	for (int i = 0; i < 100000; i++){
		read(0,buf,0);
	}
	return 0;
}
