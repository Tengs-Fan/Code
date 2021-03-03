CURRENT_DIR	:=	$(shell pwd)
BUILD_ROOT	:=	$(CURRENT_DIR)/../

IMAGE		:=	$(BUILD_ROOT)/hd32Mi.img

OBJ			:=	main

KERNEL		:=	kernel

CC			:=	clang
CFLAGS		:=	-target i386-pc-none-elf

LD			:=	x86_64-elf-ld
LDFLAGS		:=	-Ttext 0xc0001500 -m elf_i386


object:
	$(CC) $(CFLAGS) -c $(OBJ).c -o $(OBJ).o

link:
	$(LD) $(LDFLAGS) $(OBJ).o -e $(OBJ) -o $(KERNEL).bin


.PHONY: kernel clean


kernel: object link
	@dd if=$(KERNEL).bin	of=$(IMAGE) bs=512 count=200 seek=9 conv=notrunc

clean:
	rm -rf *.o *.bin 
