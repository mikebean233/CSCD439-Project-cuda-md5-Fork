All: md5kernel.cu md5.cu main.cu
	nvcc -arch=sm_3s -o -g -G md5Gpu main.cu

test: test.cu
    nvcc -arch=sm_35 -o test test.cu

clean:
	rm *.o
	rm md5
