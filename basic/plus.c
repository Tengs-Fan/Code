#include <stdio.h>


int main () {

	int i = 3;
	int k = i+++i+++i++;

	printf("the first one is %d", k);

	int j = 3;
	int p = (++j)+(++j)+(++j);

	printf("the second one is %d", p);


	}
