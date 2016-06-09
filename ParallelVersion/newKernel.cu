#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "md5.cu"
/*
 *   Maximum number of threads per multiprocessor:  2048
 *   Maximum number of threads per block:           1024
 *   Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
 *   Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
 *
 */
#define MAX_GRID_X 2147483647
#define MAX_GRID_Y 65535
#define MAX_GRID_z 65535

#define MAX_BLOCK_X 1024
#define MAX_BLOCK_Y 1024
#define MAX_BLOCK_Z 64

__device__ __constant__ int v[4];
__device__ __constant__ unsigned char charMap[26];

__global__ void crack(uint wordLength, uint beginningOffset, long long batchSize, unsigned char *out, uint charSetLength) {
    long long permutationNo = gridDim.x * blockIdx.y + blockIdx.x;

    extern __shared__ unsigned char thisWord[];

    if (permutationNo > batchSize)
        return;

    permutationNo += beginningOffset;

    int thisValue = permutationNo % (charSetLength * (threadIdx.x + 1) + 1);
    thisWord[threadIdx.x] = charMap[thisValue];
    uint c1, c2, c3, c4;
    if (permutationNo == 0){
        printf("charMap: %s\n", charMap);
        printf("v0:%d  v1:%d  v2:%d v3:%d  ", v[0], v[1], v[2], v[3]);
        //printf("%s\n", thisWord);
        //printf("permutationnNo: %lld   blockIdx.x: %d  threadIdx.x: %d\n", permutationNo, blockIdx.x, threadIdx.x);
    }
    md5_vfy(thisWord, wordLength, &c1, &c2, &c3, &c4);
    if(c1 == v[0] && c2 == v[1] && c3 == v[2] && c4 == v[3] ){
        out[threadIdx.x] = thisWord[threadIdx.x];
    }
}

void usage(char* programName);

int main(int argc, char** argv){
    // Device
    unsigned char *d_charMap, *d_out;

    // Host
    unsigned char *h_charMap, *h_out;
    //uint h_wordLength, h_batchSize, h_charSetLength;
    uint h_v[4] = {0,0,0,0};
    int inputWordLength;
    int charMapLength;


    // Configuration variables
    dim3 gridDim;
    dim3 blockDim;


    if(argc < 2)
        usage(argv[0]);

    unsigned char* inputWord = (unsigned char*) calloc(strlen(argv[1]) + 1, sizeof(unsigned char));
    strcpy((char*)inputWord, argv[1]);
    inputWordLength = strlen((const char*)inputWord);


    // Generate hash
    md5_vfy(inputWord, inputWordLength, h_v, h_v + 1, h_v + 2, h_v + 3);

    // Allocate cpu memory
    char* staticCharSet = (char*)"abcdefghijklmnopqrstuvwxyz";
    charMapLength = strlen((char*)staticCharSet);
    h_charMap = (unsigned char*) calloc(charMapLength,   sizeof(unsigned char));
    h_out     = (unsigned char*) calloc(inputWordLength, sizeof(unsigned char));
    strcpy((char*)h_charMap, (const char*) staticCharSet);


    // Allocate and initialize Gpu memory
    //cudaMalloc((void **) &d_charMap, sizeof(unsigned char) * charMapLength);
    cudaMalloc((void **) &d_out,     sizeof(unsigned char) * inputWordLength);
    cudaMemset (d_out,  0,sizeof(unsigned char) * inputWordLength);
    cudaMemset (charMap,0,sizeof(unsigned char) * charMapLength);


    cudaMemcpyToSymbol(charMap, &h_charMap, charMapLength * sizeof(unsigned char),0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(v,       &h_v,                           4 * sizeof(uint), 0, cudaMemcpyHostToDevice);


    // Calculate the number of possible permutations
    int digitNo = 1;
    long long noPermutations = charMapLength;
    for(; digitNo < inputWordLength; ++digitNo){
        noPermutations *= charMapLength;
    }

    printf("Input Word: %s\nInput Word Length: %d\nCharacter Set:\"%s\"\nPossible Permutations: $lld\n", inputWord, inputWordLength, h_charMap, noPermutations);
    int testWordLength = 1;
    for(; testWordLength <= inputWordLength; ++testWordLength){
        blockDim.x = testWordLength;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x  = (int) noPermutations;//ceil(MAX_GRID_X / testWordLength);
        gridDim.y  = 1;
        gridDim.z  = 1;

        crack <<< gridDim, blockDim, testWordLength >>> (testWordLength, 0, noPermutations, d_out, charMapLength);
        cudaMemcpy(h_out, d_out, testWordLength, cudaMemcpyDeviceToHost);
        if(h_out[0] != '\0'){
            printf("Found match: %s\n", h_out);
            break;
        }

    }
    if(h_out[0] == '\0')
        printf("No match was found :(\n");


    cudaFree(d_charMap);
    cudaFree(d_out);
}


void usage(char* programName){
    fprintf(stderr, "usage: %s testWord\n", programName);
    exit(1);
}