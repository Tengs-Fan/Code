LOG			:=	iverilog
LOGFLAGS	:=	
LOGFIX		=	v


ENV			:=	vvp
RSL			=	hola.out 


#all: %.out

%.out: %.$(LOGFIX)
	$(LOG)	$(LOGFLAGS) -o $@ $<

.PHONY: sim clean

sim: $(RSL)
	$(ENV) hola.out

clean:
	rm -rf *.out
