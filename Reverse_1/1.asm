section .bss
resb 2*32


section file1data
strHello db "Hello,World!",0ah
STRLEN equ $ - strHello

section file1text

extern print

global _start
