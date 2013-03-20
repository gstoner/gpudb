#ifndef __CPU_LIB_H__
#define __CPU_LIB_H__

#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"

static void initTable(struct tableNode * tn){
	assert(tn != NULL);
	tn->totalAttr = 0;
	tn->tupleNum = 0;
	tn->tupleSize = 0;
	tn->attrType = NULL;
	tn->attrSize = NULL;
	tn->attrTotalSize = NULL;
	tn->dataFormat = NULL;
	tn->dataPos = NULL;
	tn->content = NULL;
}

// merge the src table into the dst table. The dst table must be initialized.
// only consider the case when the data are all in the memory

static void mergeIntoTable(struct tableNode *dst, struct tableNode * src, struct statistic *pp){

	struct timespec start, end;

	clock_gettime(CLOCK_REALTIME, &start);

	assert(dst != NULL);
	assert(src != NULL);
	dst->totalAttr = src->totalAttr; 
	dst->tupleSize = src->tupleSize;
	dst->tupleNum += src->tupleNum;

	if (dst->attrType == NULL){
		dst->attrType = (int *) malloc(sizeof(int) * dst->totalAttr);
		dst->attrSize = (int *) malloc(sizeof(int) * dst->totalAttr);
		dst->attrTotalSize = (int *) malloc(sizeof(int) * dst->totalAttr);
		dst->dataPos = (int *) malloc(sizeof(int) * dst->totalAttr);
		dst->dataFormat = (int *) malloc(sizeof(int) * dst->totalAttr);

		for(int i=0;i<dst->totalAttr;i++){
			dst->attrType[i] = src->attrType[i];
			dst->attrSize[i] = src->attrSize[i];
			dst->attrTotalSize[i] = src->attrTotalSize[i];
			dst->dataPos[i] = MEM;
			dst->dataFormat[i] = src->dataFormat[i];
		}
	}

	if(dst->content == NULL){
		dst->content = (char **) malloc(sizeof(char *) * dst->totalAttr);
		for(int i=0; i<dst->totalAttr; i++){
			int size = dst->attrTotalSize[i];
			dst->content[i] = (char *) malloc(size);
			memset(dst->content[i], 0 ,size);
			if(src->dataPos[i] == MEM)
				memcpy(dst->content[i],src->content[i],size);
			else if (src->dataPos[i] == GPU)
				cudaMemcpy(dst->content[i], src->content[i],size, cudaMemcpyDeviceToHost);
		}
	}else{
		for(int i=0; i<dst->totalAttr; i++){
			dst->attrTotalSize[i] += src->attrTotalSize[i];
			int size = dst->attrTotalSize[i];
			int offset = dst->attrTotalSize[i] - src->attrTotalSize[i];
			int newSize = src->attrTotalSize[i];
			dst->content[i] = (char *) realloc(dst->content[i], size);

			if(src->dataPos[i] == MEM)
				memcpy(dst->content[i] + offset,src->content[i],newSize);
			else if (src->dataPos[i] == GPU)
				cudaMemcpy(dst->content[i] + offset, src->content[i],newSize, cudaMemcpyDeviceToHost);
		}
	}

	clock_gettime(CLOCK_REALTIME,&end);
	double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
	pp->total += timeE/(1000*1000) ;
}

static void freeTable(struct tableNode * tn){
        int i;

        for(i=0;i<tn->totalAttr;i++){
		if(tn->dataPos[i] == MEM)
                	munmap(tn->content[i], tn->attrTotalSize[i]);
		else if(tn->dataPos[i] == GPU)
			cudaFree(tn->content[i]);
		else if(tn->dataPos[i] == UVA || tn->dataPos[i] == PINNED)
			cudaFreeHost(tn->content[i]);
        }

        free(tn->attrType);
	tn->attrType = NULL;
        free(tn->attrSize);
	tn->attrSize = NULL;
	free(tn->attrTotalSize);
	tn->attrTotalSize = NULL;
	free(tn->dataFormat);
	tn->dataFormat = NULL;
	free(tn->dataPos);
	tn->dataPos = NULL;
        free(tn->content);
	tn->content = NULL;
}

static void freeScan(struct scanNode * rel){
	free(rel->whereIndex);
	rel->whereIndex = NULL;
	free(rel->outputIndex);
	rel->outputIndex = NULL;
        free(rel->filter);
	rel->filter = NULL;
	freeTable(rel->tn);
}

static void freeMathExp(struct mathExp exp){
	if (exp.exp != NULL && exp.opNum == 2){
		freeMathExp(exp.exp[0]);
		freeMathExp(exp.exp[1]);
		free(exp.exp);
		exp.exp = NULL;
	}	
}

static void freeGroupByNode(struct groupByNode * tn){
	free(tn->groupByIndex);
	tn->groupByIndex = NULL;
	for (int i=0;i<tn->outputAttrNum;i++){
		freeMathExp(tn->gbExp[i].exp);
	}
	free(tn->gbExp);
	tn->gbExp = NULL;
	freeTable(tn->table);
}

static void freeOrderByNode(struct orderByNode * tn){
	free(tn->orderBySeq);
	tn->orderBySeq = NULL;
	free(tn->orderByIndex);
	tn->orderByIndex = NULL;
	freeTable(tn->table);
}

#endif
