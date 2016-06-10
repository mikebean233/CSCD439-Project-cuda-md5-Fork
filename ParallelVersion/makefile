All: md5kernel.cu md5.cu main.cu timing.cu
	nvcc -arch=sm_35 -o md5Gpu main.cu timing.cu

test: test.cu
	nvcc -arch=sm_35 -o test test.cu

clean:
	rm *.o
	rm md5



test: md5Gpu
	echo  > output.txt
	echo "length,serial,parallel" > output.txt
	echo "1,`./md5Gpu z -s`,`./md5Gpu z`" >> output.txt
	echo "1,`./md5Gpu z -s`,`./md5Gpu z`" >> output.txt
    echo "2,`./md5Gpu zz -s`,`./md5Gpu zz`" >> output.txt
	echo "2,`./md5Gpu zz -s`,`./md5Gpu zz`" >> output.txt
    echo "3,`./md5Gpu zzz -s`,`./md5Gpu zzz`" >> output.txt
	echo "3,`./md5Gpu zzz -s`,`./md5Gpu zzz`" >> output.txt
	echo "4,`./md5Gpu zzzz -s`,`./md5Gpu zzzz`" >> output.txt
	echo "4,`./md5Gpu zzzz -s`,`./md5Gpu zzzz`" >> output.txt
    echo "5,`./md5Gpu zzzzz -s`,`./md5Gpu zzzzz`" >> output.txt
	echo "5,`./md5Gpu zzzzz -s`,`./md5Gpu zzzzz`" >> output.txt
    echo "6,`./md5Gpu zzzzzz -s`,`./md5Gpu zzzzzz`" >> output.txt
    echo "6,`./md5Gpu zzzzzz -s`,`./md5Gpu zzzzzz`" >> output.txt
	echo "7,-----,`./md5Gpu zzzzzzz`" >> output.txt
	echo "7,-----,`./md5Gpu zzzzzzz`" >> output.txt
	echo "8,-----,`./md5Gpu zzzzzzzz`" >> output.txt
	echo "8,-----,`./md5Gpu zzzzzzzz`" >> output.txt
