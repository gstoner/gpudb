#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include "common.h"

#define	HSIZE	(1024*1024)

// assume the type of the column to be encoded is int

int main(int argc, char ** argv){

	int res = 0;

	if(argc !=3 ){
		printf("Usage: dictCompression inputColumn outputColumn\n");
		exit(-1);
	}

	struct columnHeader header;
	struct dictHeader dHeader;

	int inFd, outFd;
	long size, tupleNum;
	int numOfDistinct = 0;

	inFd = open(argv[1],O_RDONLY);

	if(inFd == -1){
		printf("Failed to open the input column\n");
		exit(-1);
	}

	read(inFd, &header, sizeof(struct columnHeader));

	if(header.format != UNCOMPRESSED){
		printf("The column has already been compressed. Nested Compression not supported yet\n");
		exit(-1);
	}

	header.format = DICT;
	tupleNum = header.tupleNum;
	size = tupleNum * sizeof(int);

	char * content = (char *) malloc(size);
	if(!content){
		printf("Failed to allocate memory to accomodate the column\n");
		exit(-1);
	}

	char *table = (char *) mmap(0,size + sizeof(struct columnHeader),PROT_READ,MAP_SHARED,inFd,0);
	memcpy(content, table + sizeof(struct columnHeader), size);
	munmap(table,size + sizeof(struct columnHeader));

	close(inFd);

	int hashTable[HSIZE] ;

	memset(hashTable,-1,sizeof(int) * HSIZE);

	for(int i=0;i<tupleNum;i++){

		int key = ((int *)content)[i];
		int hKey = key % HSIZE;

		if(hashTable[hKey] == -1){
			numOfDistinct ++;
			hashTable[hKey] = key;
		}else{
			if(hashTable[hKey] == key)
				continue;

			int j = 1;
			while(hashTable[hKey] != -1 && hashTable[hKey] != key){
				hKey = key % (HSIZE + j*111) % HSIZE; 
				j = j+1;
			}

			if(hashTable[hKey] == -1){
				hashTable[hKey] = key;
				numOfDistinct ++;
			}
		}
	}

	if(numOfDistinct > MAX_DICT_NUM)
		goto END;

	int numOfBits =1 ;

	// the number of bits needed to encode all the distinct values
	while((1 << numOfBits) < numOfDistinct){
		numOfBits ++;
	}

	// align on byte
	while(numOfBits % 8 !=0)
		numOfBits ++;

	// if the number of bits is larger than the bits in an int type, donot compress
	if(numOfBits >= sizeof(int) * 8)
		goto END;

	// make sure that one int type can accomodate multple compressed values
	int stride = sizeof(int) * 8 / numOfBits;

	if(stride <= 1)
		goto END;

	dHeader.dictNum = numOfDistinct;
	dHeader.bitNum = numOfBits;

	int * result = (int *) malloc(sizeof(int) * numOfDistinct);
	if(!result){
		printf("failed to allocate memory for result hash\n");
		exit(-1);
	}

	memset(result, -1, sizeof(int) * numOfDistinct);

	for(int i=0; i<HSIZE;i++){
		if(hashTable[i] == -1)
			continue;

		int key = hashTable[i];
		int hKey = key % numOfDistinct;
		if( result[hKey] == -1){
			result[hKey] = key;
		}else{
			int j = 1;
			while(result[hKey] !=-1){
				hKey = key % (numOfDistinct + 111*j) % numOfDistinct;
				j++;
			}
			result[hKey] = key;
		}
	}

	for(int i=0;i<numOfDistinct;i++){
		dHeader.hash[i] = result[i];
	}


	int bitInInt = sizeof(int) * 8/ stride;

	outFd = open(argv[2],O_RDWR|O_CREAT);
	if(outFd == -1){
		printf("Failed to create output column\n");
		exit(-1);
	}

	write(outFd, &header, sizeof(struct columnHeader));
	write(outFd, &dHeader, sizeof(struct dictHeader));

	for(int i=0; i<tupleNum; i+= stride){

		int outInt = 0;

		for(int k=0;k<stride;k++){
			if((i+k) >= tupleNum)
				break;

			int key = ((int *)content)[k+i];
			int hKey = key % numOfDistinct;

			int j = 1;
			while(result[hKey] != key){
				hKey = key % (numOfDistinct + 111 * j) % numOfDistinct;
				j++;
			}

			hKey = hKey & 0xFFFF;
			memcpy((char *)(&outInt) + k*dHeader.bitNum/8, &hKey, dHeader.bitNum/8);

		}

		write(outFd, &outInt, sizeof(int));

	}

	close(outFd);

END:
	free(content);

	return res;
}
