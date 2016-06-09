All: md5kernel.cu md5.cu main.cu
	nvcc -arch=sm_35 -o md5Gpu main.cu

newKernel: newKernel.cu md5.o
	nvcc -arch=sm_32 -o newKernel newKernel.cu md5.o -lm

md5.o: md5.cu
	nvcc -arch=sm_32 -c -o md5.o md5.cu

clean:
	rm *.o
	rm md5
