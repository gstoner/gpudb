#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "common.h"

//currently only supports sorting of numbers
//suppose the data are already stored in column store and  binary format

struct sortObject{
	int key;
	int id;	
};

static void bubbleSort(struct sortObject *obj, int start,int num){

	for(int i=start;i<start+num-1;i++){
		struct sortObject tmp = obj[i];
		int pos = i;
		for(int j=i+1;j<start+num;j++){
			if(obj[j].key < tmp.key){
				tmp = obj[j];
				pos = j;
			}
		}
		obj[pos] = obj[i];
		obj[i] = tmp;
		
	}
}

// start to middle-1(inclusive) is the first part
// middle to end is the second part
// start, middle, and end are all array indexes

static void mergeSort(struct sortObject *obj, int start, int middle, int end){

	int firstNum = middle - start;
	int secondNum = end - middle + 1;

	if(firstNum > 1000){
		int mid = firstNum/2;
		mergeSort(obj, start, start+mid,middle-1);
	}else{
		bubbleSort(obj, start, firstNum);
	}

	if(secondNum > 1000){
		int mid = secondNum /2;
		mergeSort(obj, middle, middle+mid, end);
	}else{
		bubbleSort(obj, middle, secondNum);
	}

	struct sortObject * result = (struct sortObject *) malloc(sizeof(struct sortObject) * (end-start+1));
	if(!result){
		printf("Malloc failed in merge sort. Not enough memory.\n");
		exit(-1);
	}

	int i,j,k;
	for(i = start, j = middle, k=0; i<=middle-1 &&j<=end;){
		if(obj[i].key < obj[j].key){
			result[k++] = obj[i++];
		}else{
			result[k++] = obj[j++];
		}
	}

	while(i<=middle-1){
		result[k++] = obj[i++];
	}

	while(j<=end)
		result[k++] = obj[j++];

	memcpy(&obj[start],result, sizeof(struct sortObject)*(end-start+1));

	free(result);

}

static void primarySort(struct sortObject * obj, int num){
	int start = 0, middle = num/2, end = num-1;

	mergeSort(obj,start,middle,end);

}


// assumes all the columns from the same table stored on disk have the same prefiex(such as the table name)
// the only difference is the index 
// prerequsite: the memory should be large enough to hold all the elements from one column

int main(int argc, char **argv){

	if(argc != 4){
		printf("./primarySort inputPrefix outputPrefix index\n");
		exit(-1);
	}

	int inFd;
	int primaryIndex, largestIndex;
	struct columnHeader header;

	primaryIndex = atoi(argv[3]);
	largestIndex = 16; 

	char buf[32] = {0};

	sprintf(buf,"%s%d",argv[1],primaryIndex);

	inFd = open(buf, O_RDONLY);
	if(inFd == -1){
		printf("Failed to open the primaryIndex column\n");
		exit(-1);
	}

	read(inFd, &header ,sizeof(struct columnHeader));
	if(header.format != UNCOMPRESSED){
		printf("Cannot sort compressed data\n");
		exit(-1);
	}

	long size = header.totalTupleNum * sizeof(int);
	long tupleNum = header.totalTupleNum;
	int blockTotal = header.blockTotal;

	char * raw = (char *) malloc(size);
	if(!raw){
		printf("Malloc failed. Not enough memory\n");
		exit(-1);
	}

	long offset = sizeof(struct columnHeader);
	long inOffset = 0;
	size = header.tupleNum * sizeof(int);

	char *outTable =(char *) mmap(0,size,PROT_READ,MAP_SHARED,inFd, offset);

	memcpy(raw + inOffset, outTable, size);
        munmap(outTable,size);
	offset += size;
	inOffset += size;

	for(int i=1;i<blockTotal;i++){
		lseek(inFd, offset,SEEK_SET);

		read(inFd,&header,sizeof(struct columnHeader));
		offset += sizeof(struct columnHeader);
		size = header.tupleNum * sizeof(int); 
		outTable =(char *) mmap(0,size,PROT_READ,MAP_SHARED,inFd, offset);

		memcpy(raw + inOffset, outTable, size);
        	munmap(outTable,size);

		offset += size;
		inOffset += size;
	}

        close(inFd);

	struct sortObject * obj = (struct sortObject *) malloc(sizeof(struct sortObject ) * tupleNum);

	if(!obj){
		printf("Malloc failed. Not enough memory!\n");
		exit(-1);
	}

	for(int i=0;i<tupleNum;i++){
		obj[i].key = ((int *)raw)[i];
		obj[i].id = i;
	}

	free(raw);

	primarySort(obj,tupleNum);

	for(int i=0;i<= largestIndex;i++){

		sprintf(buf,"%s%d",argv[1],i);
		inFd = open(buf,O_RDONLY);
		if(inFd == -1){
			printf("Failed to open input column\n");
			exit(-1);
		}
		size = lseek(inFd,0,SEEK_END);
		int tupleSize = (size - blockTotal*sizeof(struct columnHeader))/tupleNum;

		raw = (char *) malloc(size);
		if(!raw){
			printf("Malloc failed when trying to write the new result. Not enough memory !\n");
			exit(-1);
		}

		inOffset = 0;
		offset = 0;

		for(int j=0;j<blockTotal;j++){
			lseek(inFd, offset, SEEK_SET);
			read(inFd, &header,sizeof(struct columnHeader));
			offset += sizeof(struct columnHeader);
			size = header.tupleNum * tupleSize;
			outTable = (char *) mmap(0,size ,PROT_READ,MAP_SHARED,inFd, offset);

			memcpy(raw+inOffset, outTable,size);
			munmap(outTable,size);

			offset += size;
			inOffset += size;
		}

		close(inFd);

		sprintf(buf,"%s%d",argv[2],i);
		int outFd = open(buf, O_RDWR|O_CREAT,S_IRWXU|S_IRUSR);
		if(outFd == -1){
			printf("Failed to create output column\n");
			exit(-1);
		}

		header.totalTupleNum = tupleNum;
		header.blockId = 0;
		header.blockTotal = blockTotal;
		header.format = UNCOMPRESSED;

		long tupleUnit, tupleRemain, tupleCount;
		tupleRemain = tupleNum;
		tupleCount = 0;

		if(tupleNum > BLOCKNUM)
			tupleUnit = BLOCKNUM;
		else
			tupleUnit = tupleNum;

		header.tupleNum = tupleUnit;
		write(outFd, &header, sizeof(struct columnHeader));

		for(int j=0;j<tupleNum;j++){

			int writeHeader = 0;
			tupleCount ++;

			if(tupleCount > BLOCKNUM){
				tupleCount = 1;
				tupleRemain -= BLOCKNUM;
				if(tupleRemain > BLOCKNUM)
					tupleUnit = BLOCKNUM;
				else
					tupleUnit = tupleRemain;
				header.tupleNum = tupleUnit;
				header.blockId ++;
				writeHeader = 1;
			}

			if(writeHeader == 1){
				write(outFd,&header, sizeof(struct columnHeader));
			}

			int id = obj[j].id;
			write(outFd, raw+id*tupleSize, tupleSize);
		}

		free(raw);
		close(outFd);
	}

	free(obj);
	return 0;
}
