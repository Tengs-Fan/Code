MBR			:=	mbr
LOADER		:=	loader
IMAGE		:=	hd32Mi.img

ASM			:=	nasm
ASMFLAGS	:=	-f elf
LDFLAGS		:=	-I include/

CC			:=	gcc
CFLAGS		:=

LINK		:=	ld
LFLAGS		:=	-m elf_i386

UNAME		:=	$(shell uname)
ifeq ($(UNAME),Linux)
	CONFIG	:=	bochsrc.wsl
	BOCHS	:=	bochs
else
	CONFIG	:=	bochsrc.mac
	BOCHS	:=	bochs
endif


.PHONY :MBR loader kernel preprocess run clean 
MBR:
	$(ASM) $(LDFLAGS) $(MBR).S -o $(MBR).bin -O0

loader: MBR
	$(ASM) $(LDFLAGS) $(LOADER).S -o $(LOADER).bin -O0

kernel: loader
	@dd if=$(MBR).bin     of=$(IMAGE) bs=512 count=1 seek=0 conv=notrunc
	@dd if=$(LOADER).bin  of=$(IMAGE) bs=512 count=4 seek=1 conv=notrunc

preprocess: kernel
	$(ASM) $(LDFLAGS)  -E $(MBR).S -o $(MBR).xpd
	$(ASM) $(LDFLAGS)  -E $(LOADER).S -o $(LOADER).xpd
	@cat -n $(MBR).xpd
	@cat -n $(LOADER).xpd

run: kernel 
	@$(BOCHS) -f $(CONFIG)

clean:
	@rm -rf *.o *.xpd *.bin *.out
