CC = gcc
CFLAGS = -O2 -Wall -Wextra
ASM = nasm
ASMFLAGS = -f elf64 -g

all: matvec_bench

matvec_scalar.o: src/matvec_scalar.asm
	$(ASM) $(ASMFLAGS) src/matvec_scalar.asm -o matvec_scalar.o

matvec_simd.o: src/matvec_simd.asm
	$(ASM) $(ASMFLAGS) src/matvec_simd.asm -o matvec_simd.o

matvec_bench: bench/main.c matvec_scalar.o matvec_simd.o
	$(CC) $(CFLAGS) bench/main.c matvec_scalar.o matvec_simd.o -lm -o matvec_bench

clean:
	rm -f *.o matvec_bench

.PHONY: all clean
