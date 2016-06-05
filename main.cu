//Salted MD5 brute force with CUDA
//By FireXware, Aug 2nd 2010.
//OSSBox.com

//TODO: rename variables so they are called length, not max, max means max size, length means length including null terimnation
//TODO: optimize
//TODO: get command line arguments
//TODO: md5 2nd block

#define MAX_BRUTE_LENGTH 14 
#define MAX_SALT_LENGTH 38
#define MAX_TOTAL (MAX_SALT_LENGTH + MAX_BRUTE_LENGTH + MAX_SALT_LENGTH)

//Performance:
#define BLOCKS 128
#define THREADS_PER_BLOCK 256
#define MD5_PER_KERNEL 600
#define OUTPUT_INTERVAL 20

__device__ __constant__ unsigned char cudaBrute[MAX_BRUTE_LENGTH];
__device__ __constant__ unsigned char cudaCharSet[95];
__device__ unsigned char correctPass[MAX_TOTAL];

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "md5.cu" //This contains our MD5 helper functions
#include "md5kernel.cu" //the CUDA thread

void checkCUDAError(const char *msg);

void ZeroFill(unsigned char* toFill, int length)
{
	int i = 0; 
	for(i = 0; i < length; i++)
		toFill[i] = 0;
}

bool BruteIncrement(unsigned char* brute, int setLen, int wordLength, int incrementBy)
{
	int i = 0;
	while(incrementBy > 0 && i < wordLength)
	{
		int add = incrementBy + brute[i];
		brute[i] = add % setLen;
		incrementBy = add / setLen;
		i++;
	}
	
	return incrementBy != 0; //we are done if there is a remainder, because we have looped over the max
}

int main( int argc, char** argv) 
{
	char* inputString = (char*) malloc(sizeof(char) * strlen(argv[1]));
	strcpy(inputString, argv[1]);
	uint v1,v2,v3,v4;
	md5_vfy((unsigned char*)inputString, strlen(inputString), &v1, &v2, &v3, &v4);
	
	cudaEvent_t launch_begin, launch_end;
	cudaEventCreate(&launch_begin);
	cudaEventCreate(&launch_end);
	
	printf("hash for %s: %#x%x%x%x\n", inputString, v1,v2,v3,v4);



//	return 0;


	//int wordLength = 7;
	int wordLength = strlen(argv[1]);
	int charSetLen = 0;




	int numThreads = BLOCKS * THREADS_PER_BLOCK;

	unsigned char currentBrute[MAX_BRUTE_LENGTH];
	unsigned char cpuCorrectPass[MAX_TOTAL];

	ZeroFill(currentBrute, MAX_BRUTE_LENGTH);
	ZeroFill(cpuCorrectPass, MAX_TOTAL);


	charSetLen = 26;
	unsigned char charSet[charSetLen];
	memcpy(charSet, "abcdefghijklmnopqrstuvwxyz", charSetLen);

	//unsigned char hash[32];
	//memcpy(hash, "f0e8fb430bbdde6ae9c879a518fd895f", 32);

	//turn the correct hash into it's four parts
	//uint v1, v2, v3, v4;
	//md5_to_ints(hash,&v1,&v2,&v3,&v4);

	//zero the container used to hold the correct pass
	cudaMemcpyToSymbol(correctPass, &cpuCorrectPass, MAX_TOTAL, 0, cudaMemcpyHostToDevice);

	//create and copy the charset to device
	cudaMemcpyToSymbol(cudaCharSet, &charSet, charSetLen, 0, cudaMemcpyHostToDevice);

	bool finished = false;
	int ct = 0;


	cudaEventRecord(launch_begin,0);

	do{


		cudaMemcpyToSymbol(cudaBrute, &currentBrute, MAX_BRUTE_LENGTH, 0, cudaMemcpyHostToDevice);
		
		//run the kernel
		dim3 dimGrid(BLOCKS);
		dim3 dimBlock(THREADS_PER_BLOCK);

		crack<<<dimGrid, dimBlock>>>(numThreads, charSetLen, wordLength, v1,v2,v3,v4);

		//get the "correct pass" and see if there really is one
		cudaMemcpyFromSymbol(&cpuCorrectPass, correctPass, MAX_TOTAL, 0, cudaMemcpyDeviceToHost);

		if(cpuCorrectPass[0] != 0)
		{
			printf("\n\nFOUND: ");
			int k = 0;
			while(cpuCorrectPass[k] != 0)
			{
				printf("%c", cpuCorrectPass[k]);
				k++;
			}
			printf("\n");
			cudaEventRecord(launch_end,0);
			cudaEventSynchronize(launch_end);
			float time = 0;
			cudaEventElapsedTime(&time, launch_begin, launch_end);

			printf("done! GPU time cost in second: %f ms\n", time / 1000);
			return 0;
		}
		
		finished = BruteIncrement(currentBrute, charSetLen, wordLength, numThreads * MD5_PER_KERNEL);

		checkCUDAError("general");
		
		if(ct % OUTPUT_INTERVAL == 0)
		{
			printf("STATUS: ");
			int k = 0;
			for(k = 0; k < wordLength; k++)
				printf("%c",charSet[currentBrute[k]]);
			printf("\n");
		}
		ct++;

		//checkCUDAError();
	} while(!finished);


	return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }                         
}
