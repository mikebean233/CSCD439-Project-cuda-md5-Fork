//Salted MD5 brute force with CUDA
//By FireXware, Aug 2nd 2010.
//OSSBox.com

//TODO: rename variables so they are called length, not max, max means max size, length means length including null terimnation
//TODO: optimize
//TODO: get command line arguments
//TODO: md5 2nd block

#define MAX_BRUTE_LENGTH 30
#define MAX_TOTAL MAX_BRUTE_LENGTH

//Performance:
#define BLOCKS 512
#define THREADS_PER_BLOCK 1024
#define MD5_PER_KERNEL 200
#define OUTPUT_INTERVAL 20

__device__ __constant__ unsigned char cudaBrute[MAX_BRUTE_LENGTH];
__device__ __constant__ unsigned char cudaCharSet[95];
__device__ unsigned char correctPass[MAX_TOTAL];

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "timing.h"

#include "md5.cu" //This contains our MD5 helper functions
#include "md5kernel.cu" //the CUDA thread

void checkCUDAError(const char *msg);
void usage(char* programName);
void performParallelSearch(unsigned char* word, unsigned char* charSet, uint wordLength, uint charSetLength, uint v1, uint v2, uint v3, uint v4, uint verbose);
void performSerialSearch(unsigned char* word, unsigned char* charSet, int wordLength, int charSetLength, uint v1, uint v2, uint v3, uint v4, int verbose);
long longPow(int base, int exponent);
int  intPow(int base, int exponent);

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

int main( int argc, char** argv) {
	cudaSetDevice(2);

	// Parse Command line arguments
	if (argc < 2)
		usage(argv[0]);

	int wordLength = strlen(argv[1]);
	int performSerial = 0;
	int verboseMode = 0;
	int paramIdx = 0;
	int found = 0;

	for (; paramIdx < argc; ++paramIdx) {
		if (strcmp(argv[paramIdx], "-v") == 0)
			verboseMode = 1;
		if (strcmp(argv[paramIdx], "-s") == 0)
			performSerial = 1;
	}

	unsigned char *inputString = (unsigned char *) malloc(sizeof(unsigned char) * wordLength + 1);
	strcpy((char *) inputString, argv[1]);
	uint v1, v2, v3, v4;

	// Generate our character set
	int charSetLength = 26;
	unsigned char charSet[charSetLength];
	memcpy(charSet, "abcdefghijklmnopqrstuvwxyz", charSetLength);

	// Generate the MD5 hash for the input data
	md5_vfy((unsigned char *) inputString, wordLength, &v1, &v2, &v3, &v4);
	if(verboseMode)
		printf("hash for %s: %#x%x%x%x\n", inputString, v1, v2, v3, v4);

	// Crack the input hash
	if (verboseMode) {
		printf("performing ");
		if (performSerial)
			printf("serial");
		else
			printf("parallel");
		printf(" brute force md5 password hash cracking...\n");
	}

	double timeBefore = currentTime();
	if (performSerial) {
		// ---------------------- CPU VERSION -----------------------------------
		if (verboseMode)
			printf("---------- Serial Version ---------------\n");

		long noCombinations = longPow(charSetLength, wordLength);
		long combinationNo, combinationsThisRound;
		int digitNo, thisGuessLength, thisDigitValue;
		uint guessV1, guessV2, guessV3, guessV4;

		unsigned char *wordGuess = (unsigned char *) calloc(sizeof(unsigned char), wordLength);
		long *powCash = (long *) calloc(sizeof(long), wordLength);
		if (wordGuess == NULL || powCash == NULL && verboseMode) {
			fprintf(stderr, "Error: Unable to allocate host memory on the heap\n");
			exit(2);
		}

		// Build our pow cash
		for (digitNo = 0; digitNo < wordLength; ++digitNo) {
			powCash[digitNo] = longPow(charSetLength, digitNo);
		}

		// Make our guesses
		thisGuessLength = 1;
		for (; thisGuessLength <= wordLength; ++thisGuessLength) {
			combinationsThisRound = longPow(charSetLength, thisGuessLength);
			combinationNo = 0;
			for (; combinationNo < combinationsThisRound; ++combinationNo) {
				for (digitNo = 0; digitNo < thisGuessLength; ++digitNo) {
					thisDigitValue = (combinationNo / powCash[digitNo]) % charSetLength;
					wordGuess[digitNo] = charSet[thisDigitValue];
				}
				md5_vfy(wordGuess, thisGuessLength, &guessV1, &guessV2, &guessV3, &guessV4);
				if (guessV1 == v1 && guessV2 == v2 && guessV3 == v3 && guessV4 == v4) {
					if (verboseMode)
						printf("FOUND: %s\n", wordGuess);
					found = 1;
					goto exit;
				}
			}
		}

		if (powCash != NULL)
			free(powCash);
		if (wordGuess != NULL)
			free(wordGuess);


		// ---------------------- CPU VERSION -----------------------------------

	}
	else {
	// ---------------------- CUDA VERSION -----------------------------------

	if (verboseMode)
		printf("---------- Parallel Version ---------------\n");

	int numThreads = BLOCKS * THREADS_PER_BLOCK;
	unsigned char currentBrute[MAX_BRUTE_LENGTH];
	unsigned char cpuCorrectPass[MAX_TOTAL];

	ZeroFill(currentBrute, MAX_BRUTE_LENGTH);
	ZeroFill(cpuCorrectPass, MAX_TOTAL);

	//zero the container used to hold the correct pass
	cudaMemcpyToSymbol(correctPass, &cpuCorrectPass, MAX_TOTAL, 0, cudaMemcpyHostToDevice);

	//create and copy the charset to device
	cudaMemcpyToSymbol(cudaCharSet, &charSet, charSetLength, 0, cudaMemcpyHostToDevice);

	bool finished = false;
	int ct = 0;

	do {
		cudaMemcpyToSymbol(cudaBrute, &currentBrute, MAX_BRUTE_LENGTH, 0, cudaMemcpyHostToDevice);

		//run the kernel
		dim3 dimGrid(BLOCKS);
		dim3 dimBlock(THREADS_PER_BLOCK);

		crack << < dimGrid, dimBlock >> > (numThreads, charSetLength, wordLength, v1, v2, v3, v4);

		//get the "correct pass" and see if there really is one
		cudaMemcpyFromSymbol(&cpuCorrectPass, correctPass, MAX_TOTAL, 0, cudaMemcpyDeviceToHost);

		if (cpuCorrectPass[0] != 0) {
			if (verboseMode) {
				printf("\n\nFOUND: ");
				int k = 0;
				while (cpuCorrectPass[k] != 0) {
					printf("%c", cpuCorrectPass[k]);
					k++;
				}
				printf("\n");
			}
			found = 1;
			goto exit;
		}

		finished = BruteIncrement(currentBrute, charSetLength, wordLength, numThreads * MD5_PER_KERNEL);

		if (ct % OUTPUT_INTERVAL == 0 && verboseMode) {
			printf("STATUS: ");
			int k = 0;

			for (k = 0; k < wordLength; k++)
				printf("%c", charSet[currentBrute[k]]);
			printf("\n");
		}
		ct++;

	} while (!finished);

	// ---------------------- CUDA VERSION -----------------------------------
	}
	exit:

	// capture the end time
	double timeAfter = currentTime();

	double timeCost = (timeAfter - timeBefore);

	if (verboseMode && found)
		printf("Time Cost: ");
	if(found)
		printf("%lf\n", timeCost);
	else
		printf("------- NOT FOUND --------\n\n");


	return 0;
}


void performSerialSearch(unsigned char* word, unsigned char* charSet, int wordLength, int charSetLength, uint v1, uint v2, uint v3, uint v4, int verbose){

}

long longPow(int base, int exponent){
	if(exponent == 0)
		return 1l;

	int result = base;
	for(int i = 1; i < exponent; ++i){
		result *= base;
	}
	return result;
}

int intPow(int base, int exponent){
	if(exponent == 0)
		return 1l;

	int result = base;
	for(int i = 1; i < exponent; ++i){
		result *= base;
	}
	return result;
}


void performParallelSearch(unsigned char* word, unsigned char* charSet, uint wordLength, uint charSetLength, uint v1, uint v2, uint v3, uint v4, uint verbose){

}

void usage(char* programName){
	fprintf(stderr, "usage: %s targetWord [-s][-v]\n", programName);
	fprintf(stderr, "       -s: perform serial Version (if omitted, the paralell version will be used)\n");
	fprintf(stderr, "       -v: Be verbose (if omitted only the time cost of the implementation will be printed\n");
	exit(1);
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
