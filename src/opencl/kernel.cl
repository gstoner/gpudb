#include "../include/common.h"

inline int stringCmp(__global char * buf1, __global char * buf2, int size){
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

inline int testCon_int(int buf1, __global char* buf2, int size, int type, int rel){
        int res = 1;
	if(rel == EQ){
                res = ( buf1 == *(((__global int*)buf2)) );
	}else if (rel == GTH){
		res = ( buf1 > *(((__global int*)buf2)) );
	}else if (rel == LTH){
		res = ( buf1 < *(((__global int*)buf2)) );
	}else if (rel == GEQ){
		res = ( buf1 >= *(((__global int*)buf2)) );
	}else if (rel == LEQ){
		res = ( buf1 <= *(((__global int*)buf2)) );
	}
	return res;
}

inline int testCon_float(float buf1, __global char* buf2, int size, int type, int rel){
        int res = 1;
	if(rel == EQ){
		res = ( buf1 == *(((__global float*)buf2)) );
	}else if (rel == GTH){
		res = ( buf1 > *(((__global float*)buf2)) );
	}else if (rel == LTH){
		res = ( buf1 < *(((__global float*)buf2)) );
	}else if (rel == GEQ){
		res = ( buf1 >= *(((__global float*)buf2)) );
	}else if (rel == LEQ){
		res = ( buf1 <= *(((__global float*)buf2)) );
	}
	return res;
}

inline int testCon_string(__global char *buf1, __global char* buf2, int size, int type, int rel){
        int res = 1;

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

        return res;
}

__kernel void cl_memset_int(__global int * ar, int num){
        size_t stride = get_global_size(0);
        size_t offset = get_global_id(0);

        for(size_t i=offset; i<num; i+= stride)
                ar[i] = 0;
}

__kernel void transform_dict_filter_init(__global int * dictFilter, __global int *fact, long tupleNum, int dNum,  __global int * filter, int byteNum){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ;

        for(size_t i=offset; i<numInt; i += stride){
                int tmp = fact[i];
                unsigned long bit = 1;

                for(int j=0; j< sizeof(int)/byteNum; j++){
                        int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
                        int fkey = (tmp & t)>> (j*byteNum*8) ;
                        filter[i* sizeof(int)/byteNum + j] = dictFilter[fkey];
                }
        }
}


__kernel void transform_dict_filter_and(__global int * dictFilter, __global int *fact, long tupleNum, int dNum,  __global int * filter, int byteNum){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ;

        for(size_t i=offset; i<numInt; i += stride){
                int tmp = fact[i];
                unsigned long bit = 1;

                for(int j=0; j< sizeof(int)/byteNum; j++){
                        int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
                        int fkey = (tmp & t)>> (j*byteNum*8) ;
                        filter[i* sizeof(int)/byteNum + j] &= dictFilter[fkey];
                }
        }
}

__kernel void transform_dict_filter_or(__global int * dictFilter, __global int *fact, long tupleNum, int dNum,  __global int * filter,int byteNum){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ;

        for(size_t i=offset; i<numInt; i += stride){
                int tmp = fact[i];
                unsigned long bit = 1;

                for(int j=0; j< sizeof(int)/byteNum; j++){
                        int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
                        int fkey = (tmp & t)>> (j*byteNum*8) ;
                        filter[i* sizeof(int)/byteNum + j] |= dictFilter[fkey];
                }
        }
}

__kernel void genScanFilter_dict_init(__global char *col, int colSize, int colType, int dNum, __global struct whereExp *where, __global int *dfilter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i=tid;i<dNum;i+=stride){
                int fkey = ((__global struct dictHeader*)col)->hash[i];
                con = testCon_int(fkey,where->content,colSize,colType,where->relation);
                dfilter[i] = con;
        }
}

__kernel void genScanFilter_dict_or(__global char *col, int colSize, int colType, int dNum, __global struct whereExp *where, __global int *dfilter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i=tid;i<dNum;i+=stride){
                int fkey = ((__global struct dictHeader*)col)->hash[i];
                con = testCon_int(fkey,where->content,colSize,colType,where->relation);
                con = testCon_int(fkey,where->content,colSize,colType,where->relation);
                dfilter[i] |= con;
        }
}

__kernel void genScanFilter_dict_and(__global char *col, int colSize, int colType, int dNum, __global struct whereExp *where, __global int *dfilter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i=tid;i<dNum;i+=stride){
                int fkey = ((__global struct dictHeader*)col)->hash[i];
                con = testCon_int(fkey,where->content,colSize,colType,where->relation);
                dfilter[i] &= con;
        }
}

__kernel void genScanFilter_rle(__global char *col, int colSize, int colType, long tupleNum, __global struct whereExp *where, int andOr, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        int dNum = ((__global struct rleHeader*)col)->dictNum;

        for(size_t i = tid; i<dNum; i += stride){
                int fkey = ((__global int *)(col+sizeof(struct rleHeader)))[i];
                int fcount = ((__global int *)(col+sizeof(struct rleHeader)))[i + dNum];
                int fpos = ((__global int *)(col+sizeof(struct rleHeader)))[i + 2*dNum];


                con = testCon_int(fkey,where->content,colSize,colType,where->relation);

		for(int k=0;k<fcount;k++){
			if(andOr == AND)
				filter[fpos+k] &= con;
			else
				filter[fpos+k] |= con;
		}

        }
}

__kernel void genScanFilter_init_int_eq(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] == where;
                filter[i] = con;
        }
}
__kernel void genScanFilter_init_float_eq(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] == where;
                filter[i] = con;
        }
}
__kernel void genScanFilter_init_int_gth(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] > where;
                filter[i] = con;
        }
}

__kernel void genScanFilter_init_float_gth(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] > where;
                filter[i] = con;
        }
}

__kernel void genScanFilter_init_int_lth(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] < where;
                filter[i] = con;
        }
}
__kernel void genScanFilter_init_float_lth(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] < where;
                filter[i] = con;
        }
}
__kernel void genScanFilter_init_int_geq(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] >= where;
                filter[i] = con;
        }
}
__kernel void genScanFilter_init_float_geq(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] >= where;
                filter[i] = con;
        }
}

__kernel void genScanFilter_init_int_leq(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] <= where;
                filter[i] = con;
        }
}
__kernel void genScanFilter_init_float_leq(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] <= where;
                filter[i] = con;
        }
}


__kernel void genScanFilter_and(__global char *col, int colSize, int  colType, long tupleNum, __global struct whereExp * where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = testCon_string(col+colSize*i,where->content,colSize,colType,where->relation);
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_int_eq(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] == where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_float_eq(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] == where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_int_geq(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] >= where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_float_geq(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] >= where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_int_leq(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] <= where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_float_leq(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] <= where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_int_gth(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] > where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_float_gth(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] > where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_int_lth(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] < where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_float_lth(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] < where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_init(__global char *col, int colSize, int  colType, long tupleNum, __global struct whereExp * where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;
        int rel = where->relation;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = testCon_string(col+colSize*i,where->content,colSize,colType, rel);
                filter[i] = con;
        }
}

__kernel void genScanFilter_or(__global char *col, int colSize, int  colType, long tupleNum, __global struct whereExp * where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;
        int rel = where->relation;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = testCon_string(col+colSize*i,where->content,colSize,colType, rel);
                filter[i] |= con;
        }
}

__kernel void genScanFilter_or_int_eq(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] == where;
                filter[i] |= con;
        }
}
__kernel void genScanFilter_or_float_eq(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] == where;
                filter[i] |= con;
        }
}
__kernel void genScanFilter_or_int_gth(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] > where;
                filter[i] |= con;
        }
}

__kernel void genScanFilter_or_float_gth(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] > where;
                filter[i] |= con;
        }
}

__kernel void genScanFilter_or_int_lth(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] < where;
                filter[i] |= con;
        }
}
__kernel void genScanFilter_or_float_lth(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] < where;
                filter[i] |= con;
        }
}
__kernel void genScanFilter_or_int_geq(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] >= where;
                filter[i] |= con;
        }
}
__kernel void genScanFilter_or_float_geq(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] >= where;
                filter[i] |= con;
        }
}

__kernel void genScanFilter_or_int_leq(__global int *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] <= where;
                filter[i] |= con;
        }
}
__kernel void genScanFilter_or_float_leq(__global float *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = col[i] <= where;
                filter[i] |= con;
        }
}

__kernel void countScanNum(__global int *filter, long tupleNum, __global int * count){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int localCount = 0;

        for(size_t i = tid; i<tupleNum; i += stride){
                localCount += filter[i];
        }

        count[tid] = localCount;

}

__kernel void scan_dict_other(__global char *col, __global char * dict, int byteNum,int colSize, long tupleNum, __global int *psum, long resultNum, __global int * filter, __global char * result){

	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        __global struct dictHeader *dheader = (__global struct dictHeader*)dict;
        int pos = psum[tid] * colSize;

        for(size_t i = tid; i<tupleNum; i+= stride){
                if(filter[i] == 1){
                        int key = 0;
			char * buf = (char *)&key;

			for(int k=0;k<dheader->bitNum/8;k++)
				buf[k] = (col + sizeof(struct dictHeader) + i* dheader->bitNum/8)[k];

			int kvalue = dheader->hash[key];
			buf = (char *) &kvalue;
			for(int k=0;k<colSize;k++)
				(result+pos)[k] = buf[k];
                        pos += colSize;
                }
        }
}

__kernel void scan_dict_int(__global char *col, __global char * dict,int byteNum,int colSize, long tupleNum, __global int *psum, long resultNum, __global int * filter, __global int * result){

	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int localCount = psum[tid];
        __global struct dictHeader *dheader = (__global struct dictHeader*)dict;

        for(size_t i = tid; i<tupleNum; i+= stride){
                if(filter[i] == 1){
                        int key = 0;
			char * buf = (char *)&key;
			for(int k=0;k<byteNum;k++)
				buf[k] = (col+i*byteNum)[k];
                        result[localCount] = dheader->hash[key];
                        localCount ++;
                }
        }
}

__kernel void scan_other(__global char *col, int colSize, long tupleNum, __global int *psum, long resultNum, __global int * filter, __global char * result){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int pos = psum[tid]  * colSize;

        for(size_t i = tid; i<tupleNum;i+=stride){

                if(filter[i] == 1){
			for(int k=0;k<colSize;k++)
				(result+pos)[k] = (col+i*colSize)[k];
                        pos += colSize;
                }
        }
}

__kernel void scan_int(__global int *col, int colSize, long tupleNum, __global int *psum, long resultNum, __global int * filter, __global int * result){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int localCount = psum[tid] ;

        for(size_t i = tid; i<tupleNum;i+=stride){

                if(filter[i] == 1){
                        result[localCount] = col[i];
                        localCount ++;
                }
        }
}

__kernel void unpack_rle(__global char * fact, __global char * rle, long tupleNum, int dNum){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        for(size_t i=offset; i<dNum; i+=stride){

                int fvalue = ((__global int *)(fact+sizeof(struct rleHeader)))[i];
                int fcount = ((__global int *)(fact+sizeof(struct rleHeader)))[i + dNum];
                int fpos = ((__global int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

		for(int k=0;k<fcount;k++){
			((__global int*)rle)[fpos+ k] = fvalue;
		}
        }
}

//The following kernels are for traditional hash joins

__kernel void count_hash_num(__global int *dim, long  inNum, __global int *num){
	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        for(size_t i=offset;i<inNum;i+=stride){
                int joinKey = dim[i];
                int hKey = joinKey & (HSIZE-1);
                atomic_add(&(num[hKey]),1);
        }
}

__kernel void build_hash_table(__global int *dim, long inNum, __global int *psum, __global int * bucket){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        for(size_t i=offset;i<inNum;i+=stride){
                int joinKey = dim[i];
                int hKey = joinKey & (HSIZE-1);
                int pos = atomic_add(&psum[hKey],1) * 2;
                bucket[pos] = joinKey;
                pos += 1;
                int dimId = i+1;
                bucket[pos] = dimId;
        }

}

__kernel void count_join_result_dict(__global int *num, __global int* psum, __global int* bucket, __global char* fact, int dNum, __global int* dictFilter){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        __global struct dictHeader *dheader = (__global struct dictHeader *) fact;

        for(size_t i=offset;i<dNum;i+=stride){
                int fkey = dheader->hash[i];
                int hkey = fkey &(HSIZE-1);
                int keyNum = num[hkey];
		int fvalue = 0;

                for(int j=0;j<keyNum;j++){
                        int pSum = psum[hkey];
                        int dimKey = bucket[2*j + 2*pSum];
                        int dimId = bucket[2*j + 2*pSum + 1];
                        if( dimKey == fkey){
				fvalue = dimId;
                                break;
                        }
                }
                dictFilter[i] = fvalue;
        }

}


__kernel void filter_count(long tupleNum, __global int * count, __global int * factFilter){

        int lcount = 0;
	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        for(size_t i=offset; i<tupleNum; i+=stride){
                if(factFilter[i] !=0)
                        lcount ++;
        }
        count[offset] = lcount;
}

__kernel void count_join_result_rle(__global int* num, __global int* psum, __global int* bucket, __global char* fact, long tupleNum, __global int * factFilter){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        __global struct rleHeader *rheader = (__global struct rleHeader *)fact;
        int dNum = rheader->dictNum;

        for(size_t i=offset; i<dNum; i += stride){
                int fkey = ((__global int *)(fact+sizeof(struct rleHeader)))[i];
                int fcount = ((__global int *)(fact+sizeof(struct rleHeader)))[i + dNum];
                int fpos = ((__global int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

                int hkey = fkey &(HSIZE-1);
                int keyNum = num[hkey];
                int pSum = psum[hkey];

                for(int j=0;j<keyNum;j++){

                        int dimKey = bucket[2*j + 2*pSum];
                        int dimId = bucket[2*j + 2*pSum + 1];

                        if( dimKey == fkey){

                                for(int k=0;k<fcount;k++)
                                        factFilter[fpos+k] = dimId;


                                break;
                        }
                }
        }

}

__kernel  void count_join_result(__global int* num, __global int* psum, __global int* bucket, __global int* fact, long inNum, __global int* count, __global int * factFilter){
        int lcount = 0;
	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        for(size_t i=offset;i<inNum;i+=stride){
                int fkey = fact[i];
                int hkey = fkey &(HSIZE-1);
                int keyNum = num[hkey];
		int fvalue = 0;

                for(int j=0;j<keyNum;j++){
                        int pSum = psum[hkey];
                        int dimKey = bucket[2*j + 2*pSum];
                        int dimId = bucket[2*j + 2*pSum + 1];
                        if( dimKey == fkey){
                                lcount ++;
				fvalue = dimId;
                                break;
                        }
                }
                factFilter[i] = fvalue;
        }

        count[offset] = lcount;
}

__kernel void rle_psum(__global int *count, __global char * fact,  long  tupleNum,  __global int * filter){

	size_t offset = get_global_id(0);
	size_t stride = get_global_size(0);

        __global struct rleHeader *rheader = (__global struct rleHeader *) fact;
        int dNum = rheader->dictNum;

        for(size_t i= offset; i<dNum; i+= stride){

                int fcount = ((__global int *)(fact+sizeof(struct rleHeader)))[i + dNum];
                int fpos = ((__global int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];
                int lcount= 0;

		for(int k=0;k<fcount;k++){
			if(filter[fpos+ k]!=0)
				lcount++;
		}
		count[i] = lcount;
        }

}

__kernel void joinFact_rle(__global int *resPsum, __global char * fact,  int attrSize, long  tupleNum, __global int * filter, __global int * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);

        __global struct rleHeader *rheader = (__global struct rleHeader *) fact;
        int dNum = rheader->dictNum;

        for(size_t i = startIndex; i<dNum; i += stride){
                int fkey = ((__global int *)(fact+sizeof(struct rleHeader)))[i];
                int fcount = ((__global int *)(fact+sizeof(struct rleHeader)))[i + dNum];
                int fpos = ((__global int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

		int toffset = resPsum[i];
		for(int j=0;j<fcount;j++){
			if(filter[fpos-j] !=0){
				result[toffset] = fkey ;
				toffset ++;
			}
		}
        }

}

__kernel void joinFact_dict_other(__global int *resPsum, __global char * fact,  __global char *dict, int byteNum,int attrSize, long  num, __global int * filter, __global char * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localOffset = resPsum[startIndex] * attrSize;

        __global struct dictHeader *dheader = (__global struct dictHeader*)dict;

        for(size_t i=startIndex;i<num;i+=stride){
                if(filter[i] != 0){
                        int key = 0;
			char *buf = (char *) &key;
			for(int k=0;k<byteNum;k++)
				buf[k] = (fact + i*byteNum)[k];
			int kvalue = dheader->hash[key];
			buf = (char *)&kvalue;
			for(int k=0;k<attrSize;k++)
				(result + localOffset)[k] = buf[k];
                        localOffset += attrSize;
                }
        }
}

__kernel void joinFact_dict_int(__global int *resPsum, __global char * fact, __global char *dict, int byteNum, int attrSize, long  num, __global int * filter, __global int * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localCount = resPsum[startIndex];

        __global struct dictHeader *dheader = (__global struct dictHeader*)dict;

        for(size_t i=startIndex;i<num;i+=stride){
                if(filter[i] != 0){
                        int key = 0;
			char *buf = (char *)&key;
			for(int k=0;k<byteNum;k++)
				buf[k] = (fact + i *byteNum)[k];
                        result[localCount] = dheader->hash[key];
                        localCount ++;
                }
        }
}

__kernel void joinFact_other(__global int *resPsum, __global char * fact,  int attrSize, long  num, __global int * filter, __global char * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localOffset = resPsum[startIndex] * attrSize;

        for(size_t i=startIndex;i<num;i+=stride){
                if(filter[i] != 0){
			for(int k=0;k<attrSize;k++)
				(result+localOffset)[k] = (fact + i*attrSize)[k];
                        localOffset += attrSize;
                }
        }
}

__kernel void joinFact_int(__global int *resPsum, __global int * fact,  int attrSize, long  num, __global int * filter, __global int * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localCount = resPsum[startIndex];

        for(size_t i=startIndex;i<num;i+=stride){
                if(filter[i] != 0){
                        result[localCount] = fact[i];
                        localCount ++;
                }
        }
}

__kernel void joinDim_rle(__global int *resPsum, __global char * dim, int attrSize, long tupleNum,__global int * filter, __global int * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localCount = resPsum[startIndex];

        __global struct rleHeader *rheader = (__global struct rleHeader *) dim;
        int dNum = rheader->dictNum;

        for(size_t i = startIndex; i<tupleNum; i += stride){
                int dimId = filter[i];
                if(dimId != 0){
                        for(int j=0;j<dNum;j++){
                                int dkey = ((__global int *)(dim+sizeof(struct rleHeader)))[j];
                                int dcount = ((__global int *)(dim+sizeof(struct rleHeader)))[j + dNum];
                                int dpos = ((__global int *)(dim+sizeof(struct rleHeader)))[j + 2*dNum];

                                if(dpos == dimId || ((dpos < dimId) && (dpos + dcount) > dimId)){
                                        result[localCount] = dkey ;
                                        localCount ++;
                                        break;
                                }

                        }
                }
        }
}

__kernel void joinDim_dict_other(__global int *resPsum, __global char * dim, __global char *dict, int byteNum, int attrSize, long num, __global int * filter, __global char * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localOffset = resPsum[startIndex] * attrSize;

        __global struct dictHeader *dheader = (__global struct dictHeader*)dict;

        for(size_t i=startIndex;i<num;i+=stride){
                int dimId = filter[i];
                if( dimId != 0){
                        int key = 0;
			char *buf = (char *)&key;
			for(int k=0;k<byteNum;k++)
				buf[k] = (dim + (dimId-1) * byteNum)[k];
			int kvalue = dheader->hash[key];
			buf = (char *)&kvalue;
			for(int k=0;k<attrSize;k++)
				(result+localOffset)[k] = buf[k];
                        localOffset += attrSize;
                }
        }
}

__kernel void joinDim_dict_int(__global int *resPsum, __global char * dim, __global char *dict, int byteNum, int attrSize, long num, __global int * filter, __global int * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localCount = resPsum[startIndex];

        __global struct dictHeader *dheader = (__global struct dictHeader*)dict;

        for(size_t i=startIndex;i<num;i+=stride){
                int dimId = filter[i];
                if( dimId != 0){
                        int key = 0;
			char * buf = (char *)&key;
			for(int k=0;k<byteNum;k++)
				buf[k] = (dim + (dimId-1)*byteNum)[k];
                        result[localCount] = dheader->hash[key];
                        localCount ++;
                }
        }
}

__kernel void joinDim_int(__global int *resPsum, __global int * dim, int attrSize, long num, __global int * filter, __global int * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localCount = resPsum[startIndex];

        for(size_t i=startIndex;i<num;i+=stride){
                int dimId = filter[i];
                if( dimId != 0){
                        result[localCount] = dim[dimId-1];
                        localCount ++;
                }
        }
}

__kernel void joinDim_other(__global int *resPsum, __global char * dim, int attrSize, long num, __global int * filter, __global char * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localOffset = resPsum[startIndex] * attrSize;

        for(size_t i=startIndex;i<num;i+=stride){
                int dimId = filter[i];
                if( dimId != 0){
			for(int k=0;k<attrSize;k++)
				(result+localOffset)[k] = (dim + (dimId-1)*attrSize)[k];
                        localOffset += attrSize;
                }
        }
}


// for groupBy

char * gpuItoa(int value, char* result, int base){

        if (base < 2 || base > 36) {
                *result = '\0';
                return result;
        }

        char* ptr = result, *ptr1 = result, tmp_char;
        int tmp_value;

        do {
                tmp_value = value;
                value /= base;
                *ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
        } while ( value );

        if (tmp_value < 0)
                *ptr++ = '-';

        *ptr-- = '\0';

        while(ptr1 < ptr) {
                tmp_char = *ptr;
                *ptr--= *ptr1;
                *ptr1++ = tmp_char;
        }
        return result;

}

char * gpuStrcpy(char * dst, const char * src){

        char * orig = dst;
        while(*src)
                *dst++ = *src++;
        *dst = '\0';

        return orig;
}

char* gpuStrncat(char *dest, const char *src, size_t n)
{
        int dest_len = 0;
        int i;

        char * tmp = dest;
        while(*tmp != '\0'){
                tmp++;
                dest_len ++;
        }

        for (i = 0 ; i < n && src[i] != '\0' ; i++)
                dest[dest_len + i] = src[i];
        dest[dest_len + i] = '\0';
        return dest;
}

char * gpuStrcat(char * dest, const char * src){
        char *tmp =dest;
        int dest_len = 0;
        int i;

        while (*tmp!= '\0'){
                tmp++ ;
                dest_len ++;
        }

        for(i=0; src[i] !='\0'; i++){
                dest[dest_len + i] = src[i];
        }

        dest[dest_len + i] = '\0';

        return dest;
}

unsigned int StringHash(const char* s)
{
    unsigned int hash = 0;
    int c;

    while((c = *s++))
    {
        hash = ((hash << 5) + hash) ^ c;
    }

    return hash;
}


__kernel void build_groupby_key(__global char * content, __global long * colOffset, int gbColNum, __global int * gbIndex, __global int * gbType, __global int * gbSize, long tupleNum, __global int * key, __global int *num){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        for(size_t i = offset; i< tupleNum; i+= stride){
                char buf[128] = {0};
                for (int j=0; j< gbColNum; j++){
                        char tbuf[32]={0};
                        int index = gbIndex[j];
			long offset = colOffset[index];

                        if (index == -1){
                                gpuItoa(1,tbuf,10);
                                gpuStrncat(buf,tbuf,1);

                        }else if (gbType[j] == STRING){
				for(int k=0;k<gbSize[j];k++)
					tbuf[k] = (content+offset+i*gbSize[j])[k];
                                gpuStrncat(buf, tbuf, gbSize[j]);

                        }else if (gbType[j] == INT){
                                int key = ((__global int *)(content+offset))[i];
                                gpuItoa(key,tbuf,10);
                                gpuStrcat(buf,tbuf);
                        }
                }
                int hkey = StringHash(buf) % HSIZE;
                key[i]= hkey;
                num[hkey] = 1;
        }
}

__kernel void count_group_num(__global int *num, int tupleNum, __global int *totalCount){
	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);
        int localCount = 0;

        for(size_t i=offset; i<tupleNum; i+= stride){
                if(num[i] == 1){
                        localCount ++;
                }
        }

        atomic_add(totalCount,localCount);
}

float getExp(__global char *content, __global int * colOffset,struct colExp exp,int pos){
	float res = 0;;
        if(exp.op == NOOP){
                if (exp.opType == CONS)
                        res = exp.opValue;
                else{
                        int index = exp.opValue;
                        res = ((__global int *)(content+colOffset[index]))[pos];
                }
	}
	return res;
}

float calMathExp(__global char *content, __global int * colOffset,struct groupByExp exp, __global struct colExp *mexp, int pos){
        float res ;

        if(exp.exp.op == NOOP){
                if (exp.exp.opType == CONS)
                        res = exp.exp.opValue;
                else{
                        int index = exp.exp.opValue;
                        res = ((__global int *)(content+colOffset[index]))[pos];
                }

        }else if(exp.exp.op == PLUS ){
                res = getExp(content,colOffset,mexp[2*pos],pos) + getExp(content, colOffset,mexp[2*pos+1],pos);

        }else if (exp.exp.op == MINUS){
                res = getExp(content,colOffset,mexp[2*pos],pos) - getExp(content, colOffset,mexp[2*pos+1],pos);

        }else if (exp.exp.op == MULTIPLY){
                res = getExp(content,colOffset,mexp[2*pos],pos) * getExp(content, colOffset,mexp[2*pos+1], pos);

        }else if (exp.exp.op == DIVIDE){
                res = getExp(content,colOffset,mexp[2*pos],pos) / getExp(content, colOffset,mexp[2*pos+1],pos);
        }

        return res;
}


// for atomic add on float type data

inline void AtomicAdd(__global float *source, float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void agg_cal_cons(__global char * content, __global int* colOffset, int colNum, __global struct groupByExp* exp, __global struct colExp *mexp, __global int * gbType, __global int * gbSize, long tupleNum, __global int * key, __global int *psum,  __global char * result, __global long * resOffset){

	size_t stride = get_global_size(0);
	size_t index = get_global_id(0);

        float buf[32];
        for(int i=0;i<32;i++)
                buf[i] = 0;

        for(size_t i=index;i<tupleNum;i+=stride){

                for(int j=0;j<colNum;j++){
                        int func = exp[j].func;
                        if (func == SUM){
                                float tmpRes = calMathExp(content, colOffset,exp[j], mexp, i);
                                buf[j] += tmpRes;
                        }
                }
        }

        for(int i=0;i<colNum;i++)
                AtomicAdd(&((__global float *)(result+resOffset[i]))[0], buf[i]);
}

__kernel void agg_cal(__global char * content, __global int *colOffset, int colNum, __global struct groupByExp* exp, __global struct colExp *mexp, __global int * gbType, __global int * gbSize, long tupleNum, __global int * key, __global int *psum,  __global char * result, __global long * resOffset){

	size_t stride = get_global_size(0);
	size_t index = get_global_id(0);

        for(int i=index;i<tupleNum;i+=stride){

                int hKey = key[i];
                int offset = psum[hKey];

                for(int j=0;j<colNum;j++){
                        int func = exp[j].func;
                        if(func ==NOOP){
                                int type = exp[j].exp.opType;

                                if(type == CONS){
                                        int value = exp[j].exp.opValue;
                                        ((__global int *)(result + resOffset[j]))[offset] = value;
                                }else{
                                        int index = exp[j].exp.opValue;
                                        int attrSize = gbSize[j];
                                        if(attrSize == sizeof(int))
                                                ((__global int *)(result+resOffset[j]))[offset] = ((__global int*)(content + colOffset[index]))[i];
                                        else{
						for(int k=0;k<attrSize;k++)
							(result+resOffset[j] + offset*attrSize)[k] = (content + colOffset[index] + i*attrSize)[k];
					}
                                }

                        }else if (func == SUM){
                                float tmpRes = calMathExp(content, colOffset, exp[j],mexp, i);
                                AtomicAdd(& ((__global float *)(result+resOffset[j]))[offset], tmpRes);
                        }
                }
        }
}


// for scan
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

inline int CONFLICT_FREE_OFFSET(int index)
{
        return ((index) >> LOG_NUM_BANKS);
}

inline void loadSharedChunkFromMem(__local int *s_data,
                                                                           __global int *g_idata,
                                                                           int n, int baseIndex,
                                                                           int* ai, int* bi,
                                                                           int* mem_ai, int* mem_bi,
                                                                           int* bankOffsetA, int* bankOffsetB, int isNP2)
{
        size_t thid = get_local_id(0);
        *mem_ai = baseIndex + thid;
        *mem_bi = *mem_ai + get_local_size(0);

        *ai = thid;
        *bi = thid + get_local_size(0);

        // compute spacing to avoid bank conflicts
        *bankOffsetA = CONFLICT_FREE_OFFSET(*ai);
        *bankOffsetB = CONFLICT_FREE_OFFSET(*bi);

        s_data[*ai + *bankOffsetA] = g_idata[*mem_ai];

        if (isNP2)
        {
                s_data[*bi + *bankOffsetB] = (*bi < n) ? g_idata[*mem_bi] : 0;
        }
        else
        {
                s_data[*bi + *bankOffsetB] = g_idata[*mem_bi];
        }
}

inline void storeSharedChunkToMem(__global int* g_odata,
                                      __local int* s_data,
                                      int n,
                                      int ai, int bi,
                                      int mem_ai, int mem_bi,
                                      int bankOffsetA, int bankOffsetB, int isNP2)
{
    barrier(CLK_LOCAL_MEM_FENCE); 

    g_odata[mem_ai] = s_data[ai + bankOffsetA];
    if (isNP2)
    {
        if (bi < n)
            g_odata[mem_bi] = s_data[bi + bankOffsetB];
    }
    else
    {
        g_odata[mem_bi] = s_data[bi + bankOffsetB];
    }
}

inline void clearLastElement(__local int* s_data,
                                 __global int *g_blockSums,
                                 int blockIndex, int storeSum)
{
    if (get_local_id(0) == 0)
    {
        int index = (get_local_size(0) << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);

        if (storeSum) 
        {
            g_blockSums[blockIndex] = s_data[index];
        }

        s_data[index] = 0;
    }
}

inline int buildSum(__local int *s_data)
{
    int thid = get_local_id(0);
    int stride = 1;

    for (size_t d = get_local_size(0); d > 0; d >>= 1)
    {
	barrier(CLK_LOCAL_MEM_FENCE);

        if (thid < d)
        {
            int i  = mul24(mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_data[bi] += s_data[ai];
        }

        stride *= 2;
    }

    return stride;
}

void scanRootToLeaves(__local int *s_data, int stride)
{
    int thid = get_local_id(0);

    for (size_t d = 1; d <= get_local_size(0); d *= 2)
    {
        stride >>= 1;

	barrier(CLK_LOCAL_MEM_FENCE);

        if (thid < d)
        {
            int i  = mul24(mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t  = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

void prescanBlock(__local int *data, int blockIndex, __global int *blockSums, int storeSum)
{
    int stride = buildSum(data);               // build the sum in place up the tree
    clearLastElement(data, blockSums,
                               (blockIndex == 0) ? get_group_id(0) : blockIndex, storeSum);
    scanRootToLeaves(data, stride);            // traverse down tree to build the scan 
}

__kernel void prescan(__global int *g_odata,
                        __global int *g_idata,
                        __global int *g_blockSums,
                        int n,
                        int blockIndex,
                        int baseIndex, int storeSum, int isNP2, __local int * s_data
                                                )
{
    int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
    int bid = get_group_id(0);
    int bsize = get_local_size(0);

    loadSharedChunkFromMem(s_data, g_idata, n,
                                  (baseIndex == 0) ?
                                  mul24(bid, (bsize << 1)):baseIndex,
                                  &ai, &bi, &mem_ai, &mem_bi,
                                  &bankOffsetA, &bankOffsetB, isNP2);

    prescanBlock(s_data, blockIndex, g_blockSums,storeSum);

    storeSharedChunkToMem(g_odata, s_data, n,
                                 ai, bi, mem_ai, mem_bi,
                                 bankOffsetA, bankOffsetB, isNP2);
}



__kernel void uniformAdd(__global int *g_data,
                           __global int *uniforms,
                           int n,
                           int blockOffset,
                           int baseIndex)
{
    __local int uni;
    if (get_local_id(0) == 0)
        uni = uniforms[get_group_id(0) + blockOffset];

    int bid = get_group_id(0);
    int bsize = get_local_size(0);

    int address = mul24(bid, (bsize << 1)) + baseIndex + get_local_id(0);

    barrier(CLK_LOCAL_MEM_FENCE);

    g_data[address]              += uni;
    g_data[address + get_local_size(0)] += (get_local_id(0) + get_local_size(0) < n) * uni;
}

// for materialization

__kernel void materialize(__global char * content, __global long * colOffset, int colNum, __global int *attrSize, long tupleNum, int tupleSize, __global char *result){
	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);

        for(size_t i=startIndex;i<tupleNum;i+=stride){
                int offset = 0;
                for(int j=0;j<colNum;j++){
                        int aSize = attrSize[j];
			for(int k=0;k<aSize;k++)
				(result + i*tupleSize + offset)[k] = (content + colOffset[j] + i*aSize)[k];
                        offset += aSize;
                }
        }
}


// for order by

#define SAMPLE_STRIDE 128
#define SHARED_SIZE_LIMIT 1024 

int gpu_strcmp(const char *s1, const char *s2, int len){
        int res = 0;

        for(int i=0;i < len;i++){
                if(s1[i]<s2[i]){
                        res = -1;
                        break;
                }else if(s1[i]>s2[i]){
                        res = 1;
                        break;
                }
        }
        return res;

}

#define W (sizeof(int) * 8)
int nextPowerOfTwo(int x){
    /*
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    */
    return 1U << (W - __clz(x - 1));
}

int iDivUp(int a, int b){
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

int getSampleCount(int dividend){
    return iDivUp(dividend, SAMPLE_STRIDE);
}

int binarySearchInInt(int val, int *data, int L, int stride, int sortDir){
    if (L == 0)
    {
        return 0;
    }

    int pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        int newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val)))
        {
            pos = newPos;
        }
    }

    return pos;
}

int binarySearchExInt(int val, int *data, int L, int stride, int sortDir){
    if (L == 0)
    {
        return 0;
    }

    int pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        int newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val)))
        {
            pos = newPos;
        }
    }

    return pos;
}

int binarySearchIn(char * val, char *data, int L, int stride, int sortDir, int keySize){
    if (L == 0)
    {
        return 0;
    }

    int pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        int newPos = umin(pos + stride, L);

        if ((sortDir && (gpu_strcmp(data+(newPos-1)*keySize,val,keySize) != 1)) || (!sortDir && (gpu_strcmp(data + (newPos-1)*keySize,val,keySize)!=-1)))
        {
            pos = newPos;
        }
    }

    return pos;
}

int binarySearchEx(char * val, char *data, int L, int stride, int sortDir, int keySize){
    if (L == 0)
    {
        return 0;
    }

    int pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        int newPos = umin(pos + stride, L);

        if ((sortDir && (gpu_strcmp(data+(newPos-1)*keySize,val,keySize) == -1)) || (!sortDir && (gpu_strcmp(data + (newPos-1)*keySize,val,keySize)==1)))
        {
            pos = newPos;
        }
    }

    return pos;
}

__kernel void generateSampleRanksKernel(
        int *d_RanksA,
        int *d_RanksB,
        char *d_SrcKey,
        int keySize,
        int stride,
        int N,
        int threadCount,
        int sortDir
)
{
    size_t pos = get_global_id(0);

    if (pos >= threadCount)
    {
        return;
    }

    const int           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_SrcKey += segmentBase * keySize;
    d_RanksA += segmentBase / SAMPLE_STRIDE;
    d_RanksB += segmentBase / SAMPLE_STRIDE;

    const int segmentElementsA = stride;
    const int segmentElementsB = umin(stride, N - segmentBase - stride);
    const int  segmentSamplesA = getSampleCount(segmentElementsA);
    const int  segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
        d_RanksA[i] = i * SAMPLE_STRIDE;
        d_RanksB[i] = binarySearchEx(
                          d_SrcKey+i * SAMPLE_STRIDE*keySize, d_SrcKey + stride*keySize,
                          segmentElementsB, nextPowerOfTwo(segmentElementsB),sortDir,keySize
                      );
    }

    if (i < segmentSamplesB)
    {
        d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;
        d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchIn(
                                                     d_SrcKey+(stride + i * SAMPLE_STRIDE)*keySize, d_SrcKey + 0,
                                                     segmentElementsA, nextPowerOfTwo(segmentElementsA),sortDir,keySize
                                                 );
    }
}


__kernel void mergeRanksAndIndicesKernel(
    __global int *d_Limits,
    __global int *d_Ranks,
    int stride,
    int N,
    int threadCount
)
{
    size_t pos = get_global_id(0); 

    if (pos >= threadCount)
    {
        return;
    }

    const int           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_Ranks  += (pos - i) * 2;
    d_Limits += (pos - i) * 2;

    const int segmentElementsA = stride;
    const int segmentElementsB = umin(stride, N - segmentBase - stride);
    const int  segmentSamplesA = getSampleCount(segmentElementsA);
    const int  segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
        int dstPos = binarySearchExInt(d_Ranks[i], d_Ranks + segmentSamplesA, segmentSamplesB, nextPowerOfTwo(segmentSamplesB),1) + i;
        d_Limits[dstPos] = d_Ranks[i];
    }

    if (i < segmentSamplesB)
    {
        int dstPos = binarySearchInInt(d_Ranks[segmentSamplesA + i], d_Ranks, segmentSamplesA, nextPowerOfTwo(segmentSamplesA),1) + i;
        d_Limits[dstPos] = d_Ranks[segmentSamplesA + i];
    }
}


void merge(
        __local char *dstKey,
        __local int *dstVal,
        __local char *srcAKey,
        __local int *srcAVal,
        __local char *srcBKey,
        __local int *srcBVal,
        int lenA,
        int nPowTwoLenA,
        int lenB,
        int nPowTwoLenB,
        int sortDir,
        int keySize
)
{
        char keyA[64], keyB[64];
        int valA, valB, dstPosA, dstPosB;

	size_t threadId = get_local_id(0);

    if ( < lenA)
    {
	for(int i=0;i<keySize;i++)
		keyA[i] = srcAKey[threadId * keySize + i];
        valA = srcAVal[threadId];
        dstPosA = binarySearchEx(keyA, srcBKey, lenB, nPowTwoLenB, sortDir,keySize) + threadId;
    }

    if (threadId < lenB)
    {
	for(int i=0;i<keySize;i++)
		keyB[i] = srcBKey[threadId * keySize + i];
        valB = srcBVal[threadId];
        dstPosB = binarySearchIn(keyB, srcAKey, lenA, nPowTwoLenA, sortDir, keySize) + threadId;
    }


    barrier(CLK_LOCAL_MEM_FENCE); 

    if (threadId < lenA)
    {
	for(int i=0;i<keySize;i++)
		dstKey[dstPosA*keySize + i] = keyA[i];
        dstVal[dstPosA] = valA;
    }

    if (threadId < lenB)
    {
	for(int i=0;i<keySize;i++)
		dstKey[dstPosB*keySize + i] = keyB[i];
        dstVal[dstPosB] = valB;
    }
}

__kernel void mergeElementaryIntervalsKernel(
        __global char *d_DstKey,
        __global int *d_DstVal,
        __global char *d_SrcKey,
        __global int *d_SrcVal,
        __global int *d_LimitsA,
        __global int *d_LimitsB,
        int stride,
        int N,
        int sortDir,
        int keySize,
	__local char *s_key,
	__local int * s_val
)
{

	size_t blockId = get_group_id(0);
	size_t threadId = get_local_id(0);

    	const int   intervalI = blockId & ((2 * stride) / SAMPLE_STRIDE - 1);
    	const int segmentBase = (blockId - intervalI) * SAMPLE_STRIDE;
    	d_SrcKey += segmentBase * keySize;
    	d_SrcVal += segmentBase;
    	d_DstKey += segmentBase * keySize;
    	d_DstVal += segmentBase;

    	__local int startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

    	if (threadId == 0){
        	int segmentElementsA = stride;
        	int segmentElementsB = umin(stride, N - segmentBase - stride);
        	int  segmentSamplesA = getSampleCount(segmentElementsA);
        	int  segmentSamplesB = getSampleCount(segmentElementsB);
        	int   segmentSamples = segmentSamplesA + segmentSamplesB;

        	startSrcA    = d_LimitsA[blockId];
        	startSrcB    = d_LimitsB[blockId];
        	int endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockId + 1] : segmentElementsA;
        	int endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockId + 1] : segmentElementsB;
        	lenSrcA      = endSrcA - startSrcA;
        	lenSrcB      = endSrcB - startSrcB;
        	startDstA    = startSrcA + startSrcB;
        	startDstB    = startDstA + lenSrcA;
    	}

    barrier(CLK_LOCAL_MEM_FENCE); 

    if (threadId < lenSrcA)
    {
        memcpy(s_key + threadId * keySize, d_SrcKey + (startSrcA + threadId)*keySize, keySize);
        s_val[threadId +             0] = d_SrcVal[0 + startSrcA + threadId];
    }

    if (threadId < lenSrcB)
    {
        memcpy(s_key + (threadId + SAMPLE_STRIDE)*keySize, d_SrcKey + (stride + startSrcB+threadId)*keySize,keySize);
        s_val[threadId + SAMPLE_STRIDE] = d_SrcVal[stride + startSrcB + threadId];
    }

    barrier(CLK_LOCAL_MEM_FENCE); 
    merge(
        s_key,
        s_val,
        s_key + 0,
        s_val + 0,
        s_key + SAMPLE_STRIDE*keySize,
        s_val + SAMPLE_STRIDE,
        lenSrcA, SAMPLE_STRIDE,
        lenSrcB, SAMPLE_STRIDE,
        sortDir,
        keySize
    );

    barrier(CLK_LOCAL_MEM_FENCE); 

    if (threadId < lenSrcA)
    {
        memcpy(d_DstKey + (startDstA + threadId)*keySize, s_key + threadId * keySize, keySize);
        d_DstVal[startDstA + threadId] = s_val[threadId];
    }

    if (threadId < lenSrcB)
    {
        memcpy(d_DstKey + (startDstB + threadId)*keySize, s_key + (lenSrcA + threadId)*keySize, keySize);
        d_DstVal[startDstB + threadId] = s_val[lenSrcA + threadId];
    }
}

__global__ static void sort_key(__global char * key, int tupleNum, int keySize, __global char *result, int *pos,int dir, __local char * bufKey, __local int* bufVal){
	size_t lid = get_local_id(0);
	size_t bid = get_group_id(0);

	size_t lsize = get_local_size(0);

        int gid = bid * SHARED_SIZE_LIMIT + lid;

	for(int i=0;i<keySize;i++)
		bufKey[lid*keySize + i] = key[gid+keySize + i];
        bufVal[lid] = gid;

	for(int i=0;i<keySize;i++)
		bufKey[i + (lid+lsize)*keySize] = key[i+(gid+lsize)*keySize];
        bufVal[lid+lsize] = gid+ lsize;

    	barrier(CLK_LOCAL_MEM_FENCE); 

        for (int size = 2; size < tupleNum && size < SHARED_SIZE_LIMIT; size <<= 1){
                int ddd = dir ^ ((lid & (size / 2)) != 0);

                for (int stride = size / 2; stride > 0; stride >>= 1){
    			barrier(CLK_LOCAL_MEM_FENCE); 
                        int pos = 2 * lid - (lid & (stride - 1));
                        Comparator(
                                bufKey+pos*keySize, bufVal[pos +      0],
                                bufKey+(pos+stride)*keySize, bufVal[pos + stride],
                                keySize,
                                ddd
                        );
                }
        }

    {
        for (int stride = blockDim.x ; stride > 0; stride >>= 1)
        {
    		barrier(CLK_LOCAL_MEM_FENCE); 
            	int pos = 2 * lid - (lid & (stride - 1));
            Comparator(
                bufKey+pos*keySize, bufVal[pos +      0],
                bufKey+(pos+stride)*keySize, bufVal[pos + stride],
                keySize,
                dir
            );
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

	for(int i=0;i<keySize;i++)
		result[i+ gid*keySize] = bufKey[lid*keySize + i];

        pos[gid] = bufVal[lid];

	for(int i=0;i<keySize;i++)
		result[i + (gid+lsize)*keySize] = bufKey[i+ (lid+lsize)*keySize];

        pos[gid+lsize] = bufVal[lid+lsize];

}

__kernel static void gather_result(__global int * keyPos, __global char * col, int newNum, int tupleNum, __global int *size, int colNum, __global char *result, __global long * offset){
	size_t stride = get_global_size(0);
	size_t index = get_global_id(0);

        for(int j=0;j<colNum;j++){
                for(size_t i=index;i<newNum;i+=stride){
                        int pos = keyPos[i];
                        if(pos<tupleNum){
				for(int k=0;k<size[j];k++)
					result[offset[j]+i*size[j]+k] = col[offset[j]+pos*size[j]+k];
			}
                }
        }
}

__kernel void build_orderby_keys(__global char * content, int tupleNum, int odNum, int keySize, __global int *index, __global int * size, __global char *key __global long* offset){
        int stride = blockDim.x * gridDim.x;
        int offset = blockIdx.x * blockDim.x + threadIdx.x;

        for(int i=offset;i<tupleNum;i+=stride){
                int pos = i* keySize;

                for(int j=0;j<odNum;j++){
			int index = index[j];
			for(int k=0;k<size[j];k++)
				key[pos + k] = content[offset[index] + i*size[j] + k];
                        pos += size[j];
                }

        }
}

__kernel void set_key(__global char *key, int tupleNum){

        size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);

        for(size_t i=tid;i<tupleNum;i+=stride)
                key[i] = '{';

}

