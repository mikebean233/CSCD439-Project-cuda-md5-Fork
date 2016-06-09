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
	// Parse Command line argumants
	if(argc < 2)
		usage(argv[0]);

	int wordLength = strlen(argv[1]);
	int performSerial = 0;
	int verboseMode = 0;
	if(argc > 2)
		if(strcmp("-s", argv[2]) == 0)
			performSerial = 1;
		else
			usage();

	if(argc > 3)
		if(strcmp("-v", argv[3]) == 0)
			verboseMode = 1;
		else
			usage();

	char* inputString = (char*) malloc(sizeof(char) * strlen(argv[1]));
	strcpy(inputString, argv[1]);
	uint v1,v2,v3,v4;

	// Generate our character set
	int charSetLen = 26;
	unsigned char charSet[charSetLen];
	memcpy(charSet, "abcdefghijklmnopqrstuvwxyz", charSetLen);

	// Generate the MD5 hash for the input data
	md5_vfy((unsigned char*)inputString, strlen(inputString), &v1, &v2, &v3, &v4);
	printf("hash for %s: %#x%x%x%x\n", inputString, v1,v2,v3,v4);

	// Crack the input hash
	if(verboseMode){
		printf("performing ");
		if(performSerial)
			printf("serial");
		else
			printf("parallel");
		printf(" brute force md5 password hash cracking...\n");
	}

	long timeBefore = clock();

	if(performSerial)
		performSerialSearch(inputString, wordLength, charSetLen, v1, v2, v3, v4, verboseMode);
	else
		performParallelSearch(intputString, wordLength, charSetLen, v1, v2, v3, v4, verboseMode);

	long timeAfter = clock();

	float timeCost = (timeBefore - timeAfter )/1000000.0;

	if(verbose)
		printf("Time Cost: ");

	printf("%f", timeCost);
	return 0;
}


void performSerialSearch(char* word, char* charset, int wordLength, int charSetLength, int v1, int v2, int v3, int v4, int verbose){
	long noCombinations = longPow(charSetLength, wordLength);
	long combinationNo, combinationsThisRound;
	int digitNo, charIdx, thisGuessLength, thisDigitValue;
	int guessV1, guessV2, guessV3, guessV4;


	char* wordGuess = (char*) calloc(sizeof(char), wordLength);
	if(wordGuess == null){
		fprintf(stderr, "Error: Unable to allocate memory to store the guessed word\n");
		exit(2);
	}

	long* powCash = (long*) calloc(sizeof(long), wordLength);

	// Build our pow cash
	int digitNo = 0;
	for (; digitNo < wordLength; ++i) {
		powCash[digitNo] = longPow(charSetLength, digitNo);
	}

	// Make our guesses
	thisGuessLength = 1;
	for (; thisGuessLength <= wordLength; ++thisGuessLength) {
		combinationsThisRound = longPow(charSetLength, thisGuessLength);
		combinationNo = 0;
		for (; combinationNo < combinationsThisRound; ++combinationNo) {
			for (digitNo = 0; digitNo < thisGuessLength) {
				thisDigitValue = (combinationNo / powCash[digitNo]) % base;
				wordGuess[digitNo] = charSet[thisDigitValue];
			}
			md5_vfy(wordGuess, thisGuessLength, &guessV1, &guessV1, &guessV2, &guessV3, &guessV4);
			if(guessV1 == v1 && guessV2 == v2 && guessV3 == v3 && guessV4 == v4){
				if(verbose)
					printf("FOUND: %s", wordGuess);
				return;
			}
		}
	}

	if(powCash != null)
		free(powCash);
	if(wordGuess != null)
		free(wordGuess);
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


void performParallelSearch(char* word, char* charset, int wordLength, int charSetLength, int v1, int v2, int v3, int v4, int verbose){
	//cudaEvent_t launch_begin, launch_end;
	//cudaEventCreate(&launch_begin);
	//cudaEventCreate(&launch_end);

	int numThreads = BLOCKS * THREADS_PER_BLOCK;
	unsigned char currentBrute[MAX_BRUTE_LENGTH];
	unsigned char cpuCorrectPass[MAX_TOTAL];

	ZeroFill(currentBrute, MAX_BRUTE_LENGTH);
	ZeroFill(cpuCorrectPass, MAX_TOTAL);

	//zero the container used to hold the correct pass
	cudaMemcpyToSymbol(correctPass, &cpuCorrectPass, MAX_TOTAL, 0, cudaMemcpyHostToDevice);

	//create and copy the charset to device
	cudaMemcpyToSymbol(cudaCharSet, &charSet, charSetLen, 0, cudaMemcpyHostToDevice);

	bool finished = false;
	int ct = 0;

	//cudaEventRecord(launch_begin,0);

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
			if(verbose){
				printf("\n\nFOUND: ");
				int k = 0;
				while(cpuCorrectPass[k] != 0)
				{	if(verbose)
						printf("%c", cpuCorrectPass[k]);
					k++;
				}
				printf("\n");
			}
			//cudaEventRecord(launch_end,0);
			//cudaEventSynchronize(launch_end);
			//float time = 0;
			//cudaEventElapsedTime(&time, launch_begin, launch_end);

			//if(verbose)
			//	printf("done! GPU time cost in seconds: ");
			//printf("%f\n", time / 1000);
			return 0;
		}

		finished = BruteIncrement(currentBrute, charSetLen, wordLength, numThreads * MD5_PER_KERNEL);

		checkCUDAError("general");

		if(ct % OUTPUT_INTERVAL == 0 && verbose)
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


}

void usage(char* programName){
	fprintf(stderr, "usage: %s targetWord [-s][-v]\n", programName);
	fprintf(stderr, "       -s: perform serial Version (if omitted, the paralell version will be used)\n");
	fprintf(srderr, "       -v: Be verbose (if omitted only the time cost of the implementation will be printed\n");
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
