#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "scanImpl.cu"
#include "common.h"
#include "gpulib.h"

__device__ static inline int stringCmp(char* buf1, char *buf2, int size){
	int i;
	int res = 0;
	for(i=0;i<size;i++){
		if(buf1[i] > buf2[i]){
			res = 1;
			break;
		}else if (buf1[i] < buf2[i]){
			res = -1;
			break;
		}
		if(buf1[i] == 0 && buf2[i] == 0)
			break;
	}
	return res;
}

__device__ static inline int testCon(char *buf1, char* buf2, int size, int type, int rel){
	int res = 1;
	if (type == INT){
		if(rel == EQ){
			res = ( *((int*)buf1) == *(((int*)buf2)) );
		}else if (rel == GTH){
			res = ( *((int*)buf1) > *(((int*)buf2)) );
		}else if (rel == LTH){
			res = ( *((int*)buf1) < *(((int*)buf2)) );
		}else if (rel == GEQ){
			res = ( *((int*)buf1) >= *(((int*)buf2)) );
		}else if (rel == LEQ){
			res = ( *((int*)buf1) <= *(((int*)buf2)) );
		}

	}else if (type == FLOAT){
		if(rel == EQ){
			res = ( *((float*)buf1) == *(((float*)buf2)) );
		}else if (rel == GTH){
			res = ( *((float*)buf1) > *(((float*)buf2)) );
		}else if (rel == LTH){
			res = ( *((float*)buf1) < *(((float*)buf2)) );
		}else if (rel == GEQ){
			res = ( *((float*)buf1) >= *(((float*)buf2)) );
		}else if (rel == LEQ){
			res = ( *((float*)buf1) <= *(((float*)buf2)) );
		}

	}else{
		int tmp = stringCmp(buf1,buf2,size);
		if(rel == EQ){
			res = (tmp == 0);
		}else if (rel == GTH){
			res = (tmp > 0);
		}else if (rel == LTH){
			res = (tmp < 0);
		}else if (rel == GEQ){
			res = (tmp >= 0);
		}else if (rel == LEQ){
			res = (tmp <= 0);
		}
	}
	return res;
}


__global__ static void transform_dict_filter_and(int * dictFilter, char *fact, long tupleNum, int dNum,  int * filter, int byteNum){

	int stride = blockDim.x * gridDim.x;
	int offset = blockIdx.x*blockDim.x + threadIdx.x;

	int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ; 

	for(long i=offset; i<numInt; i += stride){
		int tmp = ((int *)fact)[i];
		unsigned long bit = 1;

		for(int j=0; j< sizeof(int)/byteNum; j++){
			int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
			int fkey = (tmp & t)>> (j*byteNum*8) ;
			filter[i* sizeof(int)/byteNum + j] &= dictFilter[fkey];
		}
	}
}

__global__ static void transform_dict_filter_or(int * dictFilter, char *fact, long tupleNum, int dNum,  int * filter,int byteNum){

	int stride = blockDim.x * gridDim.x;
	int offset = blockIdx.x*blockDim.x + threadIdx.x;

	int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ;

	for(long i=offset; i<numInt; i += stride){
		int tmp = ((int *)fact)[i];
		unsigned long bit = 1;

		for(int j=0; j< sizeof(int)/byteNum; j++){
			int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
			int fkey = (tmp & t)>> (j*byteNum*8) ;
			filter[i* sizeof(int)/byteNum + j] |= dictFilter[fkey];
		}
	}
}

__global__ static void genScanFilter_dict_or(char *col, int colSize, int colType, int dNum, struct whereExp *where, int *dfilter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	struct dictHeader *dheader = (struct dictHeader *) col;

	for(int i=tid;i<dNum;i+=stride){
		int fkey = dheader->hash[i];
		con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);
		dfilter[i] |= con;
	}
}

__global__ static void genScanFilter_dict_and(char *col, int colSize, int colType, int dNum, struct whereExp *where, int *dfilter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	struct dictHeader *dheader = (struct dictHeader *) col;

	for(int i=tid;i<dNum;i+=stride){
		int fkey = dheader->hash[i];
		con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);
		dfilter[i] &= con;
	}
}

__global__ static void genScanFilter_rle(char *col, int colSize, int colType, long tupleNum, long tupleOffset, struct whereExp *where, int andOr, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	struct rleHeader *rheader = (struct rleHeader *) col;
	int dNum = rheader->dictNum;

	for(int i = tid; i<dNum; i += stride){
		int fkey = ((int *)(col+sizeof(struct rleHeader)))[i];
		int fcount = ((int *)(col+sizeof(struct rleHeader)))[i + dNum];
		int fpos = ((int *)(col+sizeof(struct rleHeader)))[i + 2*dNum];

		if((fcount + fpos) < tupleOffset)
			continue;

		if(fpos >= (tupleOffset + tupleNum))
			break;

		con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);
	
		if(fpos < tupleOffset){
			int tcount = fcount + fpos - tupleOffset;
			if(tcount > tupleNum)
				tcount = tupleNum;
			for(int k=0;k<tcount;k++){
				if(andOr == AND)
					filter[k] &= con;
				else
					filter[k] |= con;
			}

		}else if((fpos + fcount) > (tupleOffset + tupleNum)){
			int tcount = tupleOffset + tupleNum - fpos ;
			for(int k=0;k<tcount;k++){
				if(andOr == AND)
					filter[fpos+k-tupleOffset] &= con;
				else
					filter[fpos+k-tupleOffset] |= con;
			}
		}else{
			for(int k=0;k<fcount;k++){
				if(andOr == AND)
					filter[fpos+k-tupleOffset] &= con;
				else
					filter[fpos+k-tupleOffset] |= con;
			}

		}
	}
}


__global__ static void genScanFilter_and(char *col, int colSize, int  colType, long tupleNum, struct whereExp * where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = testCon(col+colSize*i,where->content,colSize,colType,where->relation);
		filter[i] &= con;
	}
}

__global__ static void genScanFilter_and_int_eq(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] == where; 
		filter[i] &= con;
	}
}

__global__ static void genScanFilter_and_int_geq(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] >= where; 
		filter[i] &= con;
	}
}

__global__ static void genScanFilter_and_int_leq(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] <= where; 
		filter[i] &= con;
	}
}

__global__ static void genScanFilter_and_int_gth(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] > where; 
		filter[i] &= con;
	}
}

__global__ static void genScanFilter_and_int_lth(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] < where;
		filter[i] &= con;
	}
}

__global__ static void genScanFilter_or(char *col, int colSize, int  colType, long tupleNum, struct whereExp * where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;
	int rel = where->relation;

	for(long i = tid; i<tupleNum;i+=stride){
		con = testCon(col+colSize*i,where->content,colSize,colType, rel);
		filter[i] |= con;
	}
}

__global__ static void genScanFilter_or_int_eq(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] == where; 
		filter[i] |= con;
	}
}
__global__ static void genScanFilter_or_float_eq(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] == where; 
		filter[i] |= con;
	}
}
__global__ static void genScanFilter_or_int_gth(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] > where; 
		filter[i] |= con;
	}
}

__global__ static void genScanFilter_or_float_gth(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] > where; 
		filter[i] |= con;
	}
}
__global__ static void genScanFilter_or_int_lth(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] < where; 
		filter[i] |= con;
	}
}
__global__ static void genScanFilter_or_float_lth(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] < where; 
		filter[i] |= con;
	}
}
__global__ static void genScanFilter_or_int_geq(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] >= where;
		filter[i] |= con;
	}
}
__global__ static void genScanFilter_or_float_geq(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] >= where;
		filter[i] |= con;
	}
}
__global__ static void genScanFilter_or_int_leq(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] <= where; 
		filter[i] |= con;
	}
}
__global__ static void genScanFilter_or_float_leq(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] <= where; 
		filter[i] |= con;
	}
}
__global__ static void genScanFilter(char **col, int colNum, long tupleNum, int *rel, int * where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con = 1;

	for(long i = tid; i<tupleNum;i+=stride){

		for(int j=0;j<colNum;j++){
			int value = ((int *)(col[j]))[i];
			con &= testCon((char*)&value, (char*)&where[j],sizeof(int), INT, rel[j]);
		}
		filter[i] = con;
	}
}


__global__ static void countScanNum(int *filter, long tupleNum, int * count){
	int stride = blockDim.x * gridDim.x;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int localCount = 0;

	for(long i = tid; i<tupleNum; i += stride){
		localCount += filter[i];
	}

	count[tid] = localCount;

}


__global__ static void scan_dict_other(char *col, char * dict, int byteNum,int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){

        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	struct dictHeader *dheader = (struct dictHeader*)dict;
	int pos = psum[tid] * colSize;

	for(long i = tid; i<tupleNum; i+= stride){
		if(filter[i] == 1){
			int key = 0;
			memcpy(&key, col + sizeof(struct dictHeader) + i* dheader->bitNum/8, dheader->bitNum/8);
			memcpy(result+pos,&dheader->hash[key],colSize);
			pos += colSize;
		}
	}
}

__global__ static void scan_dict_int(char *col, char * dict,int byteNum,int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){

        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int localCount = psum[tid]; 
	struct dictHeader *dheader = (struct dictHeader*)dict;

	for(long i = tid; i<tupleNum; i+= stride){
		if(filter[i] == 1){
			int key = 0;
			memcpy(&key, col + i*byteNum, byteNum);
			((int *)result)[localCount] = dheader->hash[key];
			localCount ++;
		}
	}	
}

__global__ static void scan_other(char *col, int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int pos = psum[tid]  * colSize;

	for(long i = tid; i<tupleNum;i+=stride){
		
		if(filter[i] == 1){
			memcpy(result+pos,col+i*colSize,colSize);
			pos += colSize;
		}
	}
}

__global__ static void scan_int(char *col, int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int localCount = psum[tid] ; 

	for(long i = tid; i<tupleNum;i+=stride){
		
		if(filter[i] == 1){
			((int*)result)[localCount] = ((int*)col)[i];
			localCount ++;
		}
	}
}

__global__ static void scan_all(char **col, int colNum, long tupleNum, int *psum, long resultNum, int * filter, char ** result){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int localCount = psum[tid] ; 

	for(long i = tid; i<tupleNum;i+=stride){
		
		if(filter[i] == 1){
			for(int j=0;j<colNum;j++){
				((int*)(result[j]))[localCount] = ((int*)(col[j]))[i];
			}
			localCount ++;
		}
	}
}

__global__ void static unpack_rle(char * fact, char * rle, long tupleNum, long tupleOffset, int dNum){

	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i=offset; i<dNum; i+=stride){

		int fvalue = ((int *)(fact+sizeof(struct rleHeader)))[i];
		int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
		int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

		if((fcount + fpos) < tupleOffset)
			continue;

		if(fpos >= (tupleOffset + tupleNum))
			break;

		if(fpos < tupleOffset){
			int tcount = fcount + fpos - tupleOffset;
			if(tcount > tupleNum)
				tcount = tupleNum;
			for(int k=0;k<tcount;k++){
				((int*)rle)[k] = fvalue;
			}

		}else if ((fpos + fcount) > (tupleOffset + tupleNum)){
			int tcount = tupleNum  + tupleOffset - fpos;
			for(int k=0;k<tcount;k++){
				((int*)rle)[fpos-tupleOffset + k] = fvalue;
			}

		}else{
			for(int k=0;k<fcount;k++){
				((int*)rle)[fpos-tupleOffset + k] = fvalue;
			}
		}
	}
}

/*
 * tableScan Prerequisites:
 *	1. the input data can be fit into GPU device memory
 *	2. input data are stored in host memory
 * 
 * Input:
 *	sn: contains the data to be scanned and the predicate information
 *	pp: records statistics such kernel execution time and PCIe transfer time 
 *
 * Output:
 *	A new table node
 */

struct tableNode * tableScan(struct scanNode *sn, struct statistic *pp){

	struct tableNode *res = NULL;

	res = (struct tableNode *) malloc(sizeof(struct tableNode));

	res->totalAttr = sn->tn->totalAttr;
	res->tupleSize = sn->tn->tupleSize;

	res->attrType = (int *) malloc(sizeof(int) * res->totalAttr);
	res->attrSize = (int *) malloc(sizeof(int) * res->totalAttr);
	res->attrTotalSize = (int *) malloc(sizeof(int) * res->totalAttr);
	res->attrIndex = (int *) malloc(sizeof(int) * res->totalAttr);
	res->dataPos = (int *) malloc(sizeof(int) * res->totalAttr);
	res->dataFormat = (int *) malloc(sizeof(int) * res->totalAttr);
	res->content = (char **) malloc(sizeof(char *) * res->totalAttr);

	memcpy(res->attrType, sn->tn->attrType, sizeof(int) * res->totalAttr);
	memcpy(res->attrSize, sn->tn->attrSize, sizeof(int) * res->totalAttr);

	char ** column;
	int * gpuCount;
	int * gpuFilter;
	int * gpuPsum;

	dim3 grid(1024);
	dim3 block(256);

	int blockNum = sn->tn->tupleNum / block.x + 1;

	if(blockNum<1024)
		grid = blockNum;

	int threadNum = grid.x * block.x;
	long totalTupleNum = sn->tn->tupleNum;
	int attrNum;

	attrNum = sn->whereAttrNum;
	column = (char **) malloc(attrNum * sizeof(char *));

	int * whereFree = (int *)malloc(attrNum * sizeof(int));
	int * colWherePos = (int *)malloc(sn->tn->totalAttr * sizeof(int));


	if(!column){
		printf("Failed to allocate host memory\n");
		exit(-1);
	}

	for(int i=0;i<sn->tn->totalAttr;i++)
		colWherePos[i] = -1;

	for(int i=0;i<attrNum;i++){
		whereFree[i] = 1;
		for(int j=0;j<sn->tn->totalAttr;j++){
			if(sn->whereIndex[i] == sn->tn->attrIndex[j]){
				whereFree[i] = -1;
				colWherePos[j] = i;
			}
		}
	}

	int count, *gpuTotalCount;
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuFilter,sizeof(int) * totalTupleNum));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuPsum,sizeof(int)*threadNum));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuCount,sizeof(int)*threadNum));

	CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuPsum,0,sizeof(int) * threadNum));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuTotalCount, sizeof(int)));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuTotalCount, 0 ,sizeof(int)));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuFilter,0,sizeof(int) * totalTupleNum));

	assert(sn->hasWhere !=0);
	assert(sn->filter != NULL);

	struct whereCondition *where = sn->filter;

	if(0){

		char ** gpuColumn;
		int * gpuWhere, *cpuWhere;
		int * gpuRel, *cpuRel;

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuColumn,sizeof(char *) *where->expNum));
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuWhere, sizeof(int) * where->expNum));
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuRel, sizeof(int) * where->expNum));

		cpuWhere = (int*)malloc(sizeof(int)* where->expNum);
		cpuRel = (int*)malloc(sizeof(int)* where->expNum);

		int index, prev = -1;
		for(int i=0;i<where->expNum;i++){
			index = where->exp[i].index;
			cpuWhere[i] = *(int *) (where->exp[i].content);
			cpuRel[i] = where->exp[i].relation;
			if(prev != index){
				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &column[index] , sn->whereSize[index]));
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[index], sn->content[index], sn->whereSize[index], cudaMemcpyHostToDevice));
				prev = index;
			}
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuColumn[i],&column[index], sizeof(char*), cudaMemcpyHostToDevice));
		}

		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuWhere,cpuWhere, sizeof(int)* where->expNum, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuRel,cpuRel, sizeof(int)* where->expNum, cudaMemcpyHostToDevice));

		genScanFilter<<<grid,block>>>(gpuColumn, where->expNum,totalTupleNum, gpuRel,gpuWhere,gpuFilter);


	}else{

		struct whereExp * gpuExp;
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuExp, sizeof(struct whereExp)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuExp, &where->exp[0], sizeof(struct whereExp), cudaMemcpyHostToDevice));

		int index = where->exp[0].index;
		int prev = index;
		int format = sn->whereFormat[index];

		int prevFormat = format;
		int dNum;
		int byteNum;

		int *gpuDictFilter;

		if(sn->wherePos[index] == MEM || sn->wherePos[index] == PINNED)
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &column[index], sn->whereSize[index]));

		if(format == UNCOMPRESSED){
			if(sn->wherePos[index] == MEM || sn->wherePos[index] == PINNED)
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[index], sn->content[index], sn->whereSize[index], cudaMemcpyHostToDevice));
			else if (sn->wherePos[index] == UVA)
				column[index] = sn->content[index];

			if(sn->whereAttrType[index] == INT){
				int rel = where->exp[0].relation;
				int whereValue = *((int*) where->exp[0].content);
				if(rel==EQ)
					genScanFilter_or_int_eq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
				else if(rel == GTH)
					genScanFilter_or_int_gth<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
				else if(rel == LTH)
					genScanFilter_or_int_lth<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
				else if(rel == GEQ)
					genScanFilter_or_int_geq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
				else if (rel == LEQ)
					genScanFilter_or_int_leq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);

			}else if (sn->whereAttrType[index] == FLOAT){
				int rel = where->exp[0].relation;
				float whereValue = *((int*) where->exp[0].content);
				if(rel==EQ)
					genScanFilter_or_float_eq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
				else if(rel == GTH)
					genScanFilter_or_float_gth<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
				else if(rel == LTH)
					genScanFilter_or_float_lth<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
				else if(rel == GEQ)
					genScanFilter_or_float_geq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
				else if (rel == LEQ)
					genScanFilter_or_float_leq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);

			}else
				genScanFilter_or<<<grid,block>>>(column[index],sn->whereAttrSize[index],sn->whereAttrType[index], totalTupleNum, gpuExp, gpuFilter);

		}else if(format == DICT){
			struct dictHeader * dheader = (struct dictHeader *)sn->content[index];
			dNum = dheader->dictNum;
			byteNum = dheader->bitNum/8;

			char * gpuDictHeader;
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));

			if(sn->wherePos[index] == MEM || sn->wherePos[index] == PINNED)
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[index], sn->content[index]+sizeof(struct dictHeader), sn->whereSize[index]-sizeof(struct dictHeader), cudaMemcpyHostToDevice));
			else if (sn->wherePos[index] == UVA)
				column[index] = sn->content[index] + sizeof(struct dictHeader);

			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictFilter, dNum * sizeof(int)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuDictFilter, 0 ,dNum * sizeof(int)));

			genScanFilter_dict_or<<<grid,block>>>(gpuDictHeader,sn->whereAttrSize[index],sn->whereAttrType[index],dNum, gpuExp,gpuDictFilter);
			CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

			CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

		}else if(format == RLE){

			if(sn->wherePos[index] == MEM || sn->wherePos[index] == PINNED)
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[index], sn->content[index], sn->whereSize[index], cudaMemcpyHostToDevice));
			else if (sn->wherePos[index] == UVA)
				column[index] = sn->content[index];

			genScanFilter_rle<<<grid,block>>>(column[index],sn->whereAttrSize[index],sn->whereAttrType[index], totalTupleNum, sn->offset,gpuExp, where->andOr, gpuFilter);
		}

		CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

		int dictFilter = 0;
		int dictFinal = OR;

		for(int i=1;i<where->expNum;i++){
			index = where->exp[i].index;
			format = sn->whereFormat[index];

			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuExp, &where->exp[i], sizeof(struct whereExp), cudaMemcpyHostToDevice));

			if(prev != index){
				if(prevFormat == DICT){
					if(dictFinal == OR)
						transform_dict_filter_or<<<grid,block>>>(gpuDictFilter, column[prev], totalTupleNum, dNum, gpuFilter,byteNum);
					else
						transform_dict_filter_and<<<grid,block>>>(gpuDictFilter, column[prev], totalTupleNum, dNum, gpuFilter,byteNum);

					CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());
					CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictFilter));
					dictFinal = where->andOr;
				}

				if(whereFree[prev] == 1 && (sn->wherePos[prev] == MEM || sn->wherePos[prev] == PINNED))
					CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[prev]));

				if(sn->wherePos[index] == MEM || sn->wherePos[index] == PINNED)
					CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &column[index] , sn->whereSize[index]));

				if(format == DICT){
					if(sn->wherePos[index] == MEM || sn->wherePos[index] == PINNED)
						CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[index], sn->content[index]+sizeof(struct dictHeader), sn->whereSize[index]-sizeof(struct dictHeader), cudaMemcpyHostToDevice));
					else if (sn->wherePos[index] == UVA)
						column[index] = sn->content[index] + sizeof(struct dictHeader);

					struct dictHeader * dheader = (struct dictHeader *)sn->content[index];
					dNum = dheader->dictNum;
					byteNum = dheader->bitNum/8;

					char * gpuDictHeader;
					CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
					CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));
					CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictFilter, dNum * sizeof(int)));
					CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuDictFilter, 0 ,dNum * sizeof(int)));

					genScanFilter_dict_or<<<grid,block>>>(gpuDictHeader,sn->whereAttrSize[index],sn->whereAttrType[index],dNum, gpuExp,gpuDictFilter);
					dictFilter= -1;
					CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

				}else{
					if(sn->wherePos[index] == MEM || sn->wherePos[index] == PINNED)
						CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[index], sn->content[index], sn->whereSize[index], cudaMemcpyHostToDevice));
					else if (sn->wherePos[index] == UVA)
						column[index] = sn->content[index];
				}

				prev = index;
				prevFormat = format;
			}


			if(format == UNCOMPRESSED){
				if(sn->whereAttrType[index] == INT){
					if(where->andOr == AND){
						int rel = where->exp[i].relation;
						int whereValue = *((int*) where->exp[i].content);
						if(rel==EQ)
							genScanFilter_and_int_eq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GTH)
							genScanFilter_and_int_gth<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if(rel == LTH)
							genScanFilter_and_int_lth<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GEQ)
							genScanFilter_and_int_geq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if (rel == LEQ)
							genScanFilter_and_int_leq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
					}else{
						int rel = where->exp[i].relation;
						int whereValue = *((int*) where->exp[i].content);
						if(rel==EQ)
							genScanFilter_or_int_eq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GTH)
							genScanFilter_or_int_gth<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if(rel == LTH)
							genScanFilter_or_int_lth<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GEQ)
							genScanFilter_or_int_geq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if (rel == LEQ)
							genScanFilter_or_int_leq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
					}

				} else if (sn->whereAttrType[index] == FLOAT){
					if(where->andOr == AND){
						int rel = where->exp[i].relation;
						float whereValue = *((int*) where->exp[i].content);
						if(rel==EQ)
							genScanFilter_and_float_eq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GTH)
							genScanFilter_and_float_gth<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if(rel == LTH)
							genScanFilter_and_float_lth<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GEQ)
							genScanFilter_and_float_geq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if (rel == LEQ)
							genScanFilter_and_float_leq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
					}else{
						int rel = where->exp[i].relation;
						float whereValue = *((int*) where->exp[i].content);
						if(rel==EQ)
							genScanFilter_or_float_eq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GTH)
							genScanFilter_or_float_gth<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if(rel == LTH)
							genScanFilter_or_float_lth<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GEQ)
							genScanFilter_or_float_geq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
						else if (rel == LEQ)
							genScanFilter_or_float_leq<<<grid,block>>>(column[index],totalTupleNum, whereValue, gpuFilter);
					}
				}else{
					if(where->andOr == AND)
						genScanFilter_and<<<grid,block>>>(column[index],sn->whereAttrSize[index],sn->whereAttrType[index], totalTupleNum, gpuExp, gpuFilter);
					else
						genScanFilter_or<<<grid,block>>>(column[index],sn->whereAttrSize[index],sn->whereAttrType[index], totalTupleNum, gpuExp, gpuFilter);
				}

			}else if(format == DICT){

				struct dictHeader * dheader = (struct dictHeader *)sn->content[index];
				dNum = dheader->dictNum;
				byteNum = dheader->bitNum/8;

				char * gpuDictHeader;
				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));

				if(dictFilter != -1){
					if(where->andOr == AND)
						genScanFilter_dict_and<<<grid,block>>>(gpuDictHeader,sn->whereAttrSize[index],sn->whereAttrType[index],dNum, gpuExp,gpuDictFilter);
					else
						genScanFilter_dict_or<<<grid,block>>>(gpuDictHeader,sn->whereAttrSize[index],sn->whereAttrType[index],dNum, gpuExp,gpuDictFilter);
				}
				dictFilter = 0;

				CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

			}else if (format == RLE){
				//CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[index], sn->content[index], sn->whereSize[index], cudaMemcpyHostToDevice));
				genScanFilter_rle<<<grid,block>>>(column[index],sn->whereAttrSize[index],sn->whereAttrType[index], totalTupleNum, sn->offset,gpuExp, where->andOr, gpuFilter);

			}

			CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());
		}

		if(prevFormat == DICT){
			if(dictFinal == AND)
				transform_dict_filter_and<<<grid,block>>>(gpuDictFilter, column[prev], totalTupleNum, dNum, gpuFilter, byteNum);
			else
				transform_dict_filter_or<<<grid,block>>>(gpuDictFilter, column[prev], totalTupleNum, dNum, gpuFilter, byteNum);
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictFilter));
		}
	
		if(whereFree[prev] == 1 && (sn->wherePos[prev] == MEM || sn->wherePos[prev] == PINNED))
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[prev]));

		CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuExp));

	}

	countScanNum<<<grid,block>>>(gpuFilter,totalTupleNum,gpuCount);
	CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

	scanImpl(gpuCount,threadNum,gpuPsum, pp);

	int tmp1, tmp2;

	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp1, &gpuCount[threadNum-1], sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp2, &gpuPsum[threadNum-1], sizeof(int), cudaMemcpyDeviceToHost));

	count = tmp1+tmp2;
	res->tupleNum = count;

	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuCount));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuTotalCount));


	char **result, **scanCol;

	attrNum = sn->tn->totalAttr;

	scanCol = (char **) malloc(attrNum * sizeof(char *));
	result = (char **) malloc(attrNum * sizeof(char *));
	
	for(int i=0;i<attrNum;i++){

		int pos = colWherePos[i];

		if(pos != -1){
			scanCol[i] = column[pos];
		}else{
			if(sn->tn->dataPos[i] == MEM || sn->tn->dataPos[i] == PINNED)
				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &scanCol[i] , sn->tn->attrTotalSize[i]));

			if(sn->tn->dataFormat[i] != DICT){
				if(sn->tn->dataPos[i] == MEM || sn->tn->dataPos[i] == PINNED)
					CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(scanCol[i], sn->tn->content[i], sn->tn->attrTotalSize[i], cudaMemcpyHostToDevice));
				else
					scanCol[i] = sn->tn->content[i];

			}else{
				if(sn->tn->dataPos[i] == MEM || sn->tn->dataPos[i] == PINNED)
					CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(scanCol[i], sn->tn->content[i]+sizeof(struct dictHeader), sn->tn->attrTotalSize[i]-sizeof(struct dictHeader), cudaMemcpyHostToDevice));
				else
					scanCol[i] = sn->tn->content[i] + sizeof(struct dictHeader);
			}
		}

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &result[i], count * sn->tn->attrSize[i]));
	}

	if(0){

		char ** gpuColumn;
		char ** gpuResult;
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuColumn,attrNum * sizeof(char*)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuResult,attrNum * sizeof(char*)));

		for(int i=0;i<attrNum;i++){
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuColumn[i],&scanCol[i],sizeof(char*),cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuResult[i],&result[i],sizeof(char*),cudaMemcpyHostToDevice));
		}

		scan_all<<<grid,block>>>(gpuColumn,attrNum,totalTupleNum,gpuPsum,count,gpuFilter,gpuResult);

	}else{

		for(int i=0; i<attrNum; i++){
			int format = sn->tn->dataFormat[i];
			if(format == UNCOMPRESSED){
				if (sn->tn->attrSize[i] == sizeof(int))
					scan_int<<<grid,block>>>(scanCol[i], sn->tn->attrSize[i], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);
				else
					scan_other<<<grid,block>>>(scanCol[i], sn->tn->attrSize[i], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);

			}else if(format == DICT){
				struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[i];
				int byteNum = dheader->bitNum/8;

				char * gpuDictHeader;
				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));

				if (sn->tn->attrSize[i] == sizeof(int))
					scan_dict_int<<<grid,block>>>(scanCol[i], gpuDictHeader, byteNum,sn->tn->attrSize[i], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);
				else
					scan_dict_other<<<grid,block>>>(scanCol[i], gpuDictHeader,byteNum,sn->tn->attrSize[i], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);

				CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

			}else if(format == RLE){
				int dNum = (sn->tn->attrTotalSize[i] - sizeof(struct rleHeader))/(3*sizeof(int));
				char * gpuRle;

				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuRle, totalTupleNum * sizeof(int)));

				unpack_rle<<<grid,block>>>(scanCol[i], gpuRle,totalTupleNum, sn->offset, dNum);

				CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

				scan_int<<<grid,block>>>(gpuRle, sn->tn->attrSize[i], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);

				CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuRle));
			}

		}
	}

	CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

	for(int i=0;i<attrNum;i++){

		if(sn->tn->dataPos[i] == MEM || sn->tn->dataPos[i] == PINNED)
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(scanCol[i]));

		int colSize = res->tupleNum * res->attrSize[i];

		res->attrTotalSize[i] = colSize;
		res->dataFormat[i] = UNCOMPRESSED;

		if(sn->keepInGpu == 1){
			res->dataPos[i] = GPU;
			res->content[i] = result[i];
		}else{
			res->dataPos[i] = MEM;
			res->content[i] = (char *)malloc(colSize);
			memset(res->content[i],0,colSize);
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res->content[i],result[i],colSize ,cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaFree(result[i]));
		}
	}

	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuPsum));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuFilter));

	free(column);
	free(scanCol);
	free(result);

	return res;

}


