All: md5kernel.cu md5.cu main.cu
	nvcc -arch=sm_35 -o md5Gpu main.cu

newKernel: newKernel.cu md5.cu
	nvcc -arch=sm_32 -o -G -g newKernel newKernel.cu -lm

clean:
	rm *.o
	rm md5
