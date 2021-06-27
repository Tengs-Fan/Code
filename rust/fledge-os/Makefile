TARGET	:= x86_64-unknown-linux-gnu
MODE	:= release

V		:= @

# Binutils
OBJDUMP := rust-objdump --arch-name=x86_64
OBJCOPY := rust-objcopy --binary-architecture=x86_64


$(KERNEL_BIN): kernel
	@$(OBJCOPY) $(KERNEL_ELF) --strip-all -O binary $@

kernel:
	$(V)make build
	cp src/linker
