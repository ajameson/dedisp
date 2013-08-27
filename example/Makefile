DEDISP_HOME ?= ..
include $(DEDISP_HOME)/Makefile.inc

testdedisp: testdedisp.c
	$(NVCC) -Xcompiler -Wall -O3 -I$(DEDISP_HOME)/include -L$(DEDISP_HOME)/lib -o testdedisp testdedisp.c gasdev.c ran1.c -ldedisp -lm

clean:
	rm -f testdedisp
