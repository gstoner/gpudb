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
#include "../include/common.h"
#include "../include/gpuCudaLib.h"

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

__global__ static void transform_dict_filter_init(int * dictFilter, char *fact, long tupleNum, int dNum,  int * filter,int byteNum){

	int stride = blockDim.x * gridDim.x;
	int offset = blockIdx.x*blockDim.x + threadIdx.x;

	int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ;

	for(long i=offset; i<numInt; i += stride){
		int tmp = ((int *)fact)[i];
		unsigned long bit = 1;

		for(int j=0; j< sizeof(int)/byteNum; j++){
			int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
			int fkey = (tmp & t)>> (j*byteNum*8) ;
			filter[i* sizeof(int)/byteNum + j] = dictFilter[fkey];
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

__global__ static void genScanFilter_dict_init(char *col, int colSize, int colType, int dNum, struct whereExp *where, int *dfilter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	struct dictHeader *dheader = (struct dictHeader *) col;

	for(int i=tid;i<dNum;i+=stride){
		int fkey = dheader->hash[i];
		con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);
		dfilter[i] = con;
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

__global__ static void genScanFilter_rle(char *col, int colSize, int colType, long tupleNum, struct whereExp *where, int andOr, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	struct rleHeader *rheader = (struct rleHeader *) col;
	int dNum = rheader->dictNum;

	for(int i = tid; i<dNum; i += stride){
		int fkey = ((int *)(col+sizeof(struct rleHeader)))[i];
		int fcount = ((int *)(col+sizeof(struct rleHeader)))[i + dNum];
		int fpos = ((int *)(col+sizeof(struct rleHeader)))[i + 2*dNum];

		con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);
	
		for(int k=0;k<fcount;k++){
			if(andOr == AND)
				filter[fpos+k] &= con;
			else
				filter[fpos+k] |= con;
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

//the string is stored in the format of SOA
__global__ static void genScanFilter_and_soa(char *col, int colSize, int  colType, long tupleNum, struct whereExp * where, int * filter){

        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int rel = where->relation;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		int cmp = 0;
		for(int j=0;j<colSize;j++){
			int pos = j*tupleNum + i; 
			if(col[pos] > where->content[j]){
				cmp = 1;
				break;
			}else if (col[pos] < where->content[j]){
				cmp = -1;
				break;
			}
		}

		if (rel == EQ){
			con = (cmp == 0);
		}else if(rel == LTH){
			con = (cmp <0);
		}else if(rel == GTH){
			con = (cmp >0);
		}else if (rel == LEQ){
			con = (cmp <=0);
		}else if (rel == GEQ){
			con = (cmp >=0);
		}

		filter[i] &= con;
	}
}

__global__ static void genScanFilter_init_int_eq(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] == where; 
		filter[i] = con;
	}
}
__global__ static void genScanFilter_init_float_eq(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] == where; 
		filter[i] = con;
	}
}

__global__ static void genScanFilter_init_int_gth(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] > where; 
		filter[i] = con;
	}
}

__global__ static void genScanFilter_init_float_gth(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] > where; 
		filter[i] = con;
	}
}
__global__ static void genScanFilter_init_int_lth(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] < where; 
		filter[i] = con;
	}
}
__global__ static void genScanFilter_init_float_lth(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] < where; 
		filter[i] = con;
	}
}
__global__ static void genScanFilter_init_int_geq(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] >= where;
		filter[i] = con;
	}
}
__global__ static void genScanFilter_init_float_geq(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] >= where;
		filter[i] = con;
	}
}
__global__ static void genScanFilter_init_int_leq(char *col, long tupleNum, int where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((int*)col)[i] <= where; 
		filter[i] = con;
	}
}
__global__ static void genScanFilter_init_float_leq(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] <= where; 
		filter[i] = con;
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

__global__ static void genScanFilter_and_float_eq(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] == where; 
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

__global__ static void genScanFilter_and_float_geq(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] >= where; 
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

__global__ static void genScanFilter_and_float_leq(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] <= where; 
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

__global__ static void genScanFilter_and_float_gth(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] > where; 
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

__global__ static void genScanFilter_and_float_lth(char *col, long tupleNum, float where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		con = ((float*)col)[i] < where;
		filter[i] &= con;
	}
}

__global__ static void genScanFilter_init(char *col, int colSize, int  colType, long tupleNum, struct whereExp * where, int * filter){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int con;
	int rel = where->relation;

	for(long i = tid; i<tupleNum;i+=stride){
		con = testCon(col+colSize*i,where->content,colSize,colType, rel);
		filter[i] = con;
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

//the string is stored in the format of SOA
__global__ static void genScanFilter_or_soa(char *col, int colSize, int  colType, long tupleNum, struct whereExp * where, int * filter){

        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int rel = where->relation;
	int con;

	for(long i = tid; i<tupleNum;i+=stride){
		int cmp = 0;
		for(int j=0;j<colSize;j++){
			int pos = j*tupleNum + i; 
			if(col[pos] > where->content[j]){
				cmp = 1;
				break;
			}else if (col[pos] < where->content[j]){
				cmp = -1;
				break;
			}
		}

		if (rel == EQ){
			con = (cmp == 0);
		}else if(rel == LTH){
			con = (cmp <0);
		}else if(rel == GTH){
			con = (cmp >0);
		}else if (rel == LEQ){
			con = (cmp <=0);
		}else if (rel == GEQ){
			con = (cmp >=0);
		}

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

__global__ static void scan_other_soa(char *col, int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){
        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tNum = psum[tid];

	for(long i = tid; i<tupleNum;i+=stride){
		
		if(filter[i] == 1){
			for(int j=0;j<colSize;j++){
				long inPos = j*tupleNum + i;
				long outPos = j*resultNum + tNum;
				result[outPos] = col[inPos];
			}
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

__global__ void static unpack_rle(char * fact, char * rle, long tupleNum, int dNum){

	int offset = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i=offset; i<dNum; i+=stride){

		int fvalue = ((int *)(fact+sizeof(struct rleHeader)))[i];
		int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
		int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

		for(int k=0;k<fcount;k++){
			((int*)rle)[fpos+ k] = fvalue;
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

	struct timespec start,end;
        clock_gettime(CLOCK_REALTIME,&start);

	struct tableNode *res = NULL;
	int tupleSize = 0;

	res = (struct tableNode *) malloc(sizeof(struct tableNode));

	res->totalAttr = sn->outputNum;

	res->attrType = (int *) malloc(sizeof(int) * res->totalAttr);
	res->attrSize = (int *) malloc(sizeof(int) * res->totalAttr);
	res->attrTotalSize = (int *) malloc(sizeof(int) * res->totalAttr);
	res->attrIndex = (int *) malloc(sizeof(int) * res->totalAttr);
	res->dataPos = (int *) malloc(sizeof(int) * res->totalAttr);
	res->dataFormat = (int *) malloc(sizeof(int) * res->totalAttr);
	res->content = (char **) malloc(sizeof(char *) * res->totalAttr);

	for(int i=0;i<res->totalAttr;i++){
		int index = sn->outputIndex[i];
		res->attrType[i] = sn->tn->attrType[index];
		res->attrSize[i] = sn->tn->attrSize[index];
	}

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
	int * colWherePos = (int *)malloc(sn->outputNum * sizeof(int));


	if(!column || !whereFree || !colWherePos){
		printf("Failed to allocate host memory\n");
		exit(-1);
	}

	for(int i=0;i<sn->outputNum;i++)
		colWherePos[i] = -1;

	for(int i=0;i<attrNum;i++){
		whereFree[i] = 1;
		for(int j=0;j<sn->outputNum;j++){
			if(sn->whereIndex[i] == sn->outputIndex[j]){
				whereFree[i] = -1;
				colWherePos[j] = i;
			}
		}
	}

	int count;
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuFilter,sizeof(int) * totalTupleNum));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuPsum,sizeof(int)*threadNum));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuCount,sizeof(int)*threadNum));

	assert(sn->hasWhere !=0);
	assert(sn->filter != NULL);

	struct whereCondition *where = sn->filter;

	if(1){

		struct whereExp * gpuExp;
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuExp, sizeof(struct whereExp)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuExp, &where->exp[0], sizeof(struct whereExp), cudaMemcpyHostToDevice));

		int whereIndex = where->exp[0].index;
		int index = sn->whereIndex[whereIndex];
		int prevWhere = whereIndex;
		int prevIndex = index;

		int format = sn->tn->dataFormat[index];

		int prevFormat = format;
		int dNum;
		int byteNum;

		int *gpuDictFilter;

		if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &column[whereIndex], sn->tn->attrTotalSize[index]));

		if(format == UNCOMPRESSED){
			if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
			else if (sn->tn->dataPos[index] == UVA)
				column[whereIndex] = sn->tn->content[index];

			if(sn->tn->attrType[index] == INT){
				int rel = where->exp[0].relation;
				int whereValue = *((int*) where->exp[0].content);
				if(rel==EQ)
					genScanFilter_init_int_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
				else if(rel == GTH)
					genScanFilter_init_int_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
				else if(rel == LTH)
					genScanFilter_init_int_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
				else if(rel == GEQ)
					genScanFilter_init_int_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
				else if (rel == LEQ)
					genScanFilter_init_int_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);

			}else if (sn->tn->attrType[index] == FLOAT){
				int rel = where->exp[0].relation;
				float whereValue = *((int*) where->exp[0].content);
				if(rel==EQ)
					genScanFilter_init_float_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
				else if(rel == GTH)
					genScanFilter_init_float_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
				else if(rel == LTH)
					genScanFilter_init_float_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
				else if(rel == GEQ)
					genScanFilter_init_float_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
				else if (rel == LEQ)
					genScanFilter_init_float_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);

			}else
				genScanFilter_init<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],sn->tn->attrType[index], totalTupleNum, gpuExp, gpuFilter);

		}else if(format == DICT){
			struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
			dNum = dheader->dictNum;
			byteNum = dheader->bitNum/8;

			char * gpuDictHeader;
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));

			if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index]+sizeof(struct dictHeader), sn->tn->attrTotalSize[index]-sizeof(struct dictHeader), cudaMemcpyHostToDevice));
			else if (sn->tn->dataPos[index] == UVA)
				column[whereIndex] = sn->tn->content[index] + sizeof(struct dictHeader);

			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictFilter, dNum * sizeof(int)));

			genScanFilter_dict_init<<<grid,block>>>(gpuDictHeader,sn->tn->attrSize[index],sn->tn->attrType[index],dNum, gpuExp,gpuDictFilter);
			CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

			CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

		}else if(format == RLE){

			if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
			else if (sn->tn->dataPos[index] == UVA)
				column[whereIndex] = sn->tn->content[index];

			genScanFilter_rle<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],sn->tn->attrType[index], totalTupleNum, gpuExp, where->andOr, gpuFilter);
		}

		CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

		int dictFilter = 0;
		int dictFinal = OR;
		int dictInit = 1;

		for(int i=1;i<where->expNum;i++){
			whereIndex = where->exp[i].index;
			index = sn->whereIndex[whereIndex];
			format = sn->tn->dataFormat[index];

			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuExp, &where->exp[i], sizeof(struct whereExp), cudaMemcpyHostToDevice));

			if(prevIndex != index){
				if(prevFormat == DICT){
					if(dictInit == 1){
						transform_dict_filter_init<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter,byteNum);
						dictInit = 0;
					}else if(dictFinal == OR)
						transform_dict_filter_or<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter,byteNum);
					else
						transform_dict_filter_and<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter,byteNum);

					CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());
					CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictFilter));
					dictFinal = where->andOr;
				}

				if(whereFree[prevWhere] == 1 && (sn->tn->dataPos[prevIndex] == MEM || sn->tn->dataPos[prevIndex] == PINNED))
					CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[prevWhere]));

				if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
					CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &column[whereIndex] , sn->tn->attrTotalSize[index]));

				if(format == DICT){
					if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
						CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index]+sizeof(struct dictHeader), sn->tn->attrTotalSize[index]-sizeof(struct dictHeader), cudaMemcpyHostToDevice));
					else if (sn->tn->dataPos[index] == UVA)
						column[whereIndex] = sn->tn->content[index] + sizeof(struct dictHeader);

					struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
					dNum = dheader->dictNum;
					byteNum = dheader->bitNum/8;

					char * gpuDictHeader;
					CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
					CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));
					CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictFilter, dNum * sizeof(int)));

					genScanFilter_dict_init<<<grid,block>>>(gpuDictHeader,sn->tn->attrSize[index],sn->tn->attrType[index],dNum, gpuExp,gpuDictFilter);
					dictFilter= -1;
					CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

				}else{
					if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
						CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
					else if (sn->tn->dataPos[index] == UVA)
						column[whereIndex] = sn->tn->content[index];
				}

				prevIndex = index;
				prevWhere = whereIndex;
				prevFormat = format;
			}


			if(format == UNCOMPRESSED){
				if(sn->tn->attrType[index] == INT){
					if(where->andOr == AND){
						int rel = where->exp[i].relation;
						int whereValue = *((int*) where->exp[i].content);
						if(rel==EQ)
							genScanFilter_and_int_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GTH)
							genScanFilter_and_int_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if(rel == LTH)
							genScanFilter_and_int_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GEQ)
							genScanFilter_and_int_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if (rel == LEQ)
							genScanFilter_and_int_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
					}else{
						int rel = where->exp[i].relation;
						int whereValue = *((int*) where->exp[i].content);
						if(rel==EQ)
							genScanFilter_or_int_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GTH)
							genScanFilter_or_int_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if(rel == LTH)
							genScanFilter_or_int_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GEQ)
							genScanFilter_or_int_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if (rel == LEQ)
							genScanFilter_or_int_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
					}

				} else if (sn->tn->attrType[index] == FLOAT){
					if(where->andOr == AND){
						int rel = where->exp[i].relation;
						float whereValue = *((int*) where->exp[i].content);
						if(rel==EQ)
							genScanFilter_and_float_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GTH)
							genScanFilter_and_float_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if(rel == LTH)
							genScanFilter_and_float_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GEQ)
							genScanFilter_and_float_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if (rel == LEQ)
							genScanFilter_and_float_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
					}else{
						int rel = where->exp[i].relation;
						float whereValue = *((int*) where->exp[i].content);
						if(rel==EQ)
							genScanFilter_or_float_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GTH)
							genScanFilter_or_float_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if(rel == LTH)
							genScanFilter_or_float_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if(rel == GEQ)
							genScanFilter_or_float_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
						else if (rel == LEQ)
							genScanFilter_or_float_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
					}
				}else{
					if(where->andOr == AND)
						genScanFilter_and<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],sn->tn->attrType[index], totalTupleNum, gpuExp, gpuFilter);
					else
						genScanFilter_or<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],sn->tn->attrType[index], totalTupleNum, gpuExp, gpuFilter);
				}

			}else if(format == DICT){

				struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
				dNum = dheader->dictNum;
				byteNum = dheader->bitNum/8;

				char * gpuDictHeader;
				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));

				if(dictFilter != -1){
					if(where->andOr == AND)
						genScanFilter_dict_and<<<grid,block>>>(gpuDictHeader,sn->tn->attrSize[index],sn->tn->attrType[index],dNum, gpuExp,gpuDictFilter);
					else
						genScanFilter_dict_or<<<grid,block>>>(gpuDictHeader,sn->tn->attrSize[index],sn->tn->attrType[index],dNum, gpuExp,gpuDictFilter);
				}
				dictFilter = 0;

				CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

			}else if (format == RLE){
				//CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[index], sn->content[index], sn->whereSize[index], cudaMemcpyHostToDevice));
				genScanFilter_rle<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],sn->tn->attrType[index], totalTupleNum, gpuExp, where->andOr, gpuFilter);

			}

			CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());
		}

		if(prevFormat == DICT){
			if (dictInit == 1){
				transform_dict_filter_init<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter, byteNum);
				dictInit = 0;
			}else if(dictFinal == AND)
				transform_dict_filter_and<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter, byteNum);
			else
				transform_dict_filter_or<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter, byteNum);
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictFilter));
		}
	
		if(whereFree[prevWhere] == 1 && (sn->tn->dataPos[prevIndex] == MEM || sn->tn->dataPos[prevIndex] == PINNED))
			CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[prevWhere]));

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
	printf("scanNum %d\n",count);

	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuCount));

	char **result, **scanCol;

	attrNum = sn->outputNum;

	scanCol = (char **) malloc(attrNum * sizeof(char *));
	result = (char **) malloc(attrNum * sizeof(char *));
	
	for(int i=0;i<attrNum;i++){

		int pos = colWherePos[i];
		int index = sn->outputIndex[i];
		tupleSize += sn->tn->attrSize[index];

		if(pos != -1){
			scanCol[i] = column[pos];
		}else{
			if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &scanCol[i] , sn->tn->attrTotalSize[index]));

			if(sn->tn->dataFormat[index] != DICT){
				if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
					CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(scanCol[i], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
				else
					scanCol[i] = sn->tn->content[index];

			}else{
				if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
					CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(scanCol[i], sn->tn->content[index]+sizeof(struct dictHeader), sn->tn->attrTotalSize[index]-sizeof(struct dictHeader), cudaMemcpyHostToDevice));
				else
					scanCol[i] = sn->tn->content[index] + sizeof(struct dictHeader);
			}
		}

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &result[i], count * sn->tn->attrSize[index]));
	}

	if(1){

		for(int i=0; i<attrNum; i++){
			int index = sn->outputIndex[i];
			int format = sn->tn->dataFormat[index];
			if(format == UNCOMPRESSED){
				if (sn->tn->attrSize[index] == sizeof(int))
					scan_int<<<grid,block>>>(scanCol[i], sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);
				else
					scan_other<<<grid,block>>>(scanCol[i], sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);

			}else if(format == DICT){
				struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
				int byteNum = dheader->bitNum/8;

				char * gpuDictHeader;
				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));

				if (sn->tn->attrSize[i] == sizeof(int))
					scan_dict_int<<<grid,block>>>(scanCol[i], gpuDictHeader, byteNum,sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);
				else
					scan_dict_other<<<grid,block>>>(scanCol[i], gpuDictHeader,byteNum,sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);

				CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

			}else if(format == RLE){
				int dNum = (sn->tn->attrTotalSize[index] - sizeof(struct rleHeader))/(3*sizeof(int));
				char * gpuRle;

				CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuRle, totalTupleNum * sizeof(int)));

				unpack_rle<<<grid,block>>>(scanCol[i], gpuRle,totalTupleNum, dNum);

				CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

				scan_int<<<grid,block>>>(gpuRle, sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);

				CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuRle));
			}

		}
	}

	CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

	res->tupleSize = tupleSize;

	for(int i=0;i<attrNum;i++){

		int index = sn->outputIndex[i];

		if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
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

	clock_gettime(CLOCK_REALTIME,&end);
        double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
        printf("TableScan Time: %lf\n", timeE/(1000*1000));

	return res;

}


