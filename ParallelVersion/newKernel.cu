#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void usage(char* programName);
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

int main(int argc, char** argv){
    if(argc < 2)
        usage(argv[0]);

    char* inputWord = (char*) calloc(strlen(argv[1]), sizeof(char));
    strcpy(inputWord, argv[1]);

    printf("%s\n", inputWord);

}


void usage(char* programName){
    fprintf(stderr, "usage: %s testWord\n", programName);
    exit(1);
}