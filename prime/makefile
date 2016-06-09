CC=gcc
CFLAGS=-I.
DEPS = timing.h prime.h 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

findPrime: timing.o prime.o main.o
	gcc -o findPrime -lm -lpthread timing.o prime.o main.o -I.

clean:
	rm -r *.o findPrime
