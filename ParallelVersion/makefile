All: md5kernel.cu md5.cu main.cu timing.c
	nvcc -arch=sm_32 -g -G -o md5Gpu main.cu timing.c

test: test.cu
    nvcc -arch=sm_35 -o test test.cu

clean:
	rm *.o
	rm md5
