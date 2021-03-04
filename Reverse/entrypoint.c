#include<stdio.h>
#include<stdlib.h>

int Blabla(){

	char* string = "Hola_Mundo!";
	int length = 12;

	printf("Can it work?");
/*
	asm(	"int $0x80\n\t"
				:
				:"a"(4), "b"(1), "c"(string), "d"(length)
	   );
*/
	//exit
	//asm(	"movq $1,rax;"
		//		"xorq	%rbx,%rbx;"
		//		"int $0x80"
		// );
	exit(0);

}

int main(){
	
	printf("not called.\n");

}


	
