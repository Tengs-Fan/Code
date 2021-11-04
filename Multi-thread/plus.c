#include "threads.h"

#define n 100000000
long sum = 0;

void do_sum() { for (int i = 0; i < n; i++) sum++; }
void print()  { printf("sum = %ld\n", sum); }

int main() {
  create(do_sum);
  create(do_sum);
  join(print);
}
