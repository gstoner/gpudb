#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "common.h"

//the type of the column must be integer and the data should be stored in binary and column stored format

int main(int argc, char ** argv){

	if(argc != 3){
		printf("./rleCompresssion inputColumn outputColumn\n");
		exit(-1);
	}

	int inFd = open(argv[1],O_RDONLY);
	if(inFd == -1){
		printf("Failed to open input file\n");
		exit(-1);
	}

	struct columnHeader header;
	read(inFd, &header, sizeof(struct columnHeader));
	long tupleNum = header.tupleNum;

        long size = lseek(inFd,sizeof(struct columnHeader),SEEK_END);
        char *content = (char *) malloc(size);
        char *table =(char *) mmap(0,size + sizeof(struct columnHeader),PROT_READ,MAP_SHARED,inFd,0);
        memcpy(content,table + sizeof(struct columnHeader),size);
        munmap(table,size + sizeof(struct columnHeader));
        close(inFd);

	int outFd = open(argv[2],O_RDWR|O_CREAT);
	if(outFd == -1){
		printf("Failed to create output column\n");
		exit(-1);
	}

	struct rleHeader rheader;
	header.format = RLE;

	write(outFd, &header, sizeof(struct columnHeader));

	int distinct = 1;

	int prev = ((int *)content)[0], curr;

	for(long i=1;i<tupleNum;i++){
		curr = ((int *)content)[i];
		if(curr == prev){
			continue;
		}
		distinct ++;
		prev = curr;
	}

	int * dictValue = (int *)malloc(sizeof(int) * distinct);
	int * dictCount = (int *)malloc(sizeof(int) * distinct);
	int * dictPos = (int *)malloc(sizeof(int) * distinct);

	if(!dictPos || !dictCount || !dictValue){
		printf("Failed to allocate memory\n");
		exit(-1);
	}

	prev = ((int *)content)[0];
	int count = 1;
	int pos = 0;

	int k=0;
	for(long i =1; i<tupleNum; i++){
		curr = ((int *)content)[i];
		if(curr == prev){
			count ++;
			continue;
		}

		dictValue[k] = prev;
		dictPos[k] = pos;
		dictCount[k] = count;

		pos += count;
		k++;
		prev = curr;
		count = 1;
	}

	rheader.dictNum = distinct;

	dictValue[k] = prev;
	dictPos[k] = pos;
	dictCount[k] = count;

	write(outFd, &rheader, sizeof(struct rleHeader));

	write(outFd, dictValue, sizeof(int)*distinct);
	write(outFd, dictCount, sizeof(int)*distinct);
	write(outFd, dictPos, sizeof(int)*distinct);

	close(outFd);

	return 0;

}
