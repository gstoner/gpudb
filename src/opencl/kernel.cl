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
                res = ( buf1 == *(((int*)buf2)) );
	}else if (rel == GTH){
		res = ( buf1 > *(((int*)buf2)) );
	}else if (rel == LTH){
		res = ( buf1 < *(((int*)buf2)) );
	}else if (rel == GEQ){
		res = ( buf1 >= *(((int*)buf2)) );
	}else if (rel == LEQ){
		res = ( buf1 <= *(((int*)buf2)) );
	}
	return res;
}

inline int testCon_float(float buf1, __global char* buf2, int size, int type, int rel){
        int res = 1;
	if(rel == EQ){
		res = ( buf1 == *(((float*)buf2)) );
	}else if (rel == GTH){
		res = ( buf1 > *(((float*)buf2)) );
	}else if (rel == LTH){
		res = ( buf1 < *(((float*)buf2)) );
	}else if (rel == GEQ){
		res = ( buf1 >= *(((float*)buf2)) );
	}else if (rel == LEQ){
		res = ( buf1 <= *(((float*)buf2)) );
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

__kernel void transform_dict_filter_and(__global int * dictFilter, __global char *fact, long tupleNum, int dNum,  __global int * filter, int byteNum){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ;

        for(size_t i=offset; i<numInt; i += stride){
                int tmp = ((int *)fact)[i];
                unsigned long bit = 1;

                for(int j=0; j< sizeof(int)/byteNum; j++){
                        int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
                        int fkey = (tmp & t)>> (j*byteNum*8) ;
                        filter[i* sizeof(int)/byteNum + j] &= dictFilter[fkey];
                }
        }
}

__kernel void transform_dict_filter_or(__global int * dictFilter, __global char *fact, long tupleNum, int dNum,  __global int * filter,int byteNum){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ;

        for(size_t i=offset; i<numInt; i += stride){
                int tmp = ((int *)fact)[i];
                unsigned long bit = 1;

                for(int j=0; j< sizeof(int)/byteNum; j++){
                        int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
                        int fkey = (tmp & t)>> (j*byteNum*8) ;
                        filter[i* sizeof(int)/byteNum + j] |= dictFilter[fkey];
                }
        }
}

__kernel void genScanFilter_dict_or(__global char *col, int colSize, int colType, int dNum, __global struct whereExp *where, __global int *dfilter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        struct dictHeader *dheader = (struct dictHeader *) col;

        for(size_t i=tid;i<dNum;i+=stride){
                int fkey = dheader->hash[i];
                con = testCon_int(fkey,where->content,colSize,colType,where->relation);
                dfilter[i] |= con;
        }
}

__kernel void genScanFilter_dict_and(__global char *col, int colSize, int colType, int dNum, __global struct whereExp *where, __global int *dfilter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        struct dictHeader *dheader = (struct dictHeader *) col;

        for(size_t i=tid;i<dNum;i+=stride){
                int fkey = dheader->hash[i];
                con = testCon_int(fkey,where->content,colSize,colType,where->relation);
                dfilter[i] &= con;
        }
}

__kernel void genScanFilter_rle(__global char *col, int colSize, int colType, long tupleNum, long tupleOffset, __global struct whereExp *where, int andOr, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        struct rleHeader *rheader = (struct rleHeader *) col;
        int dNum = rheader->dictNum;

        for(size_t i = tid; i<dNum; i += stride){
                int fkey = ((int *)(col+sizeof(struct rleHeader)))[i];
                int fcount = ((int *)(col+sizeof(struct rleHeader)))[i + dNum];
                int fpos = ((int *)(col+sizeof(struct rleHeader)))[i + 2*dNum];

                if((fcount + fpos) < tupleOffset)
                        continue;

                if(fpos >= (tupleOffset + tupleNum))
                        break;

                con = testCon_int(fkey,where->content,colSize,colType,where->relation);

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


__kernel void genScanFilter_and(__global char *col, int colSize, int  colType, long tupleNum, __global struct whereExp * where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = testCon_string(col+colSize*i,where->content,colSize,colType,where->relation);
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_int_eq(__global char *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((int*)col)[i] == where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_float_eq(__global char *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((float*)col)[i] == where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_int_geq(__global char *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((int*)col)[i] >= where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_float_geq(__global char *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((float*)col)[i] >= where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_int_leq(__global char *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((int*)col)[i] <= where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_float_leq(__global char *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((float*)col)[i] <= where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_int_gth(__global char *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((int*)col)[i] > where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_float_gth(__global char *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((float*)col)[i] > where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_int_lth(__global char *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((int*)col)[i] < where;
                filter[i] &= con;
        }
}

__kernel void genScanFilter_and_float_lth(__global char *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((float*)col)[i] < where;
                filter[i] &= con;
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

__kernel void genScanFilter_or_int_eq(__global char *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((int*)col)[i] == where;
                filter[i] |= con;
        }
}
__kernel void genScanFilter_or_float_eq(__global char *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((float*)col)[i] == where;
                filter[i] |= con;
        }
}
__kernel void genScanFilter_or_int_gth(__global char *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((int*)col)[i] > where;
                filter[i] |= con;
        }
}

__kernel void genScanFilter_or_float_gth(__global char *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((float*)col)[i] > where;
                filter[i] |= con;
        }
}

__kernel void genScanFilter_or_int_lth(__global char *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((int*)col)[i] < where;
                filter[i] |= con;
        }
}
__kernel void genScanFilter_or_float_lth(__global char *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((float*)col)[i] < where;
                filter[i] |= con;
        }
}
__kernel void genScanFilter_or_int_geq(__global char *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((int*)col)[i] >= where;
                filter[i] |= con;
        }
}
__kernel void genScanFilter_or_float_geq(__global char *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((float*)col)[i] >= where;
                filter[i] |= con;
        }
}

__kernel void genScanFilter_or_int_leq(__global char *col, long tupleNum, int where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((int*)col)[i] <= where;
                filter[i] |= con;
        }
}
__kernel void genScanFilter_or_float_leq(__global char *col, long tupleNum, float where, __global int * filter){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int con;

        for(size_t i = tid; i<tupleNum;i+=stride){
                con = ((float*)col)[i] <= where;
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
        struct dictHeader *dheader = (struct dictHeader*)dict;
        int pos = psum[tid] * colSize;

        for(size_t i = tid; i<tupleNum; i+= stride){
                if(filter[i] == 1){
                        int key = 0;
			char * buf = (char *)&key;

			for(int k=0;k<dheader->bitNum/8;k++)
				buf[k] = (col + sizeof(struct dictHeader) + i* dheader->bitNum/8)[k];

			buf = (char *) &dheader->hash[key];
			for(int k=0;k<colSize;k++)
				(result+pos)[k] = buf[k];
                        pos += colSize;
                }
        }
}

__kernel void scan_dict_int(__global char *col, __global char * dict,int byteNum,int colSize, long tupleNum, __global int *psum, long resultNum, __global int * filter, __global char * result){

	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int localCount = psum[tid];
        struct dictHeader *dheader = (struct dictHeader*)dict;

        for(size_t i = tid; i<tupleNum; i+= stride){
                if(filter[i] == 1){
                        int key = 0;
			char * buf = (char *)&key;
			for(int k=0;k<byteNum;k++)
				buf[k] = (col+i*byteNum)[k];
                        ((int *)result)[localCount] = dheader->hash[key];
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

__kernel void scan_int(__global char *col, int colSize, long tupleNum, __global int *psum, long resultNum, __global int * filter, __global char * result){
	size_t stride = get_global_size(0);
	size_t tid = get_global_id(0);
        int localCount = psum[tid] ;

        for(size_t i = tid; i<tupleNum;i+=stride){

                if(filter[i] == 1){
                        ((int*)result)[localCount] = ((int*)col)[i];
                        localCount ++;
                }
        }
}

__kernel void unpack_rle(__global char * fact, __global char * rle, long tupleNum, long tupleOffset, int dNum){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        for(size_t i=offset; i<dNum; i+=stride){

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

//The following kernels are for traditional hash joins

__kernel void count_hash_num(__global char *dim, long  inNum, __global int *num){
	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        for(size_t i=offset;i<inNum;i+=stride){
                int joinKey = ((int *)dim)[i];
                int hKey = joinKey & (HSIZE-1);
                atomic_add(&(num[hKey]),1);
        }
}

__kernel void build_hash_table(__global char *dim, long inNum, __global int *psum, __global char * bucket){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        for(size_t i=offset;i<inNum;i+=stride){
                int joinKey = ((int *) dim)[i];
                int hKey = joinKey & (HSIZE-1);
                int pos = atomic_add(&psum[hKey],1) * 2;
                ((int*)bucket)[pos] = joinKey;
                pos += 1;
                int dimId = i+1;
                ((int*)bucket)[pos] = dimId;
        }

}

__kernel void count_join_result_dict(__global int *num, __global int* psum, __global char* bucket, __global char* fact, int dNum, __global int* dictFilter){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        struct dictHeader *dheader;
        dheader = (struct dictHeader *) fact;

        for(size_t i=offset;i<dNum;i+=stride){
                int fkey = dheader->hash[i];
                int hkey = fkey &(HSIZE-1);
                int keyNum = num[hkey];

                for(int j=0;j<keyNum;j++){
                        int pSum = psum[hkey];
                        int dimKey = ((int *)(bucket))[2*j + 2*pSum];
                        int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];
                        if( dimKey == fkey){
                                dictFilter[i] = dimId;
                                break;
                        }
                }
        }

}

__kernel void transform_dict_filter(__global int * dictFilter, __global char *fact, long tupleNum, int dNum,  __global int * filter){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        struct dictHeader *dheader;
        dheader = (struct dictHeader *) fact;

        int byteNum = dheader->bitNum/8;
        int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int)  ;

        for(size_t i=offset; i<numInt; i += stride){
                int tmp = ((int *)(fact + sizeof(struct dictHeader)))[i];

                for(int j=0; j< sizeof(int)/byteNum; j++){
                        int fkey = 0;
			char *buf = (char *)&fkey;
			for(int k=0;k<byteNum;k++)
				buf[k] = (((char *)&tmp) + j*byteNum)[k];

                        filter[i* sizeof(int)/byteNum + j] = dictFilter[fkey];
                }
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

__kernel void count_join_result_rle(__global int* num, __global int* psum, __global char* bucket, __global char* fact, long tupleNum, long tupleOffset,  __global int * factFilter){

	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        struct rleHeader *rheader = (struct rleHeader *)fact;
        int dNum = rheader->dictNum;

        for(size_t i=offset; i<dNum; i += stride){
                int fkey = ((int *)(fact+sizeof(struct rleHeader)))[i];
                int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
                int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

                if((fcount + fpos) < tupleOffset)
                        continue;

                if(fpos >= (tupleOffset + tupleNum))
                        break;

                int hkey = fkey &(HSIZE-1);
                int keyNum = num[hkey];

                for(int j=0;j<keyNum;j++){

                        int pSum = psum[hkey];
                        int dimKey = ((int *)(bucket))[2*j + 2*pSum];
                        int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];

                        if( dimKey == fkey){

                                if(fpos < tupleOffset){
                                        int tcount = fcount + fpos - tupleOffset;
                                        if(tcount > tupleNum)
                                                tcount = tupleNum;
                                        for(int k=0;k<tcount;k++)
                                                factFilter[k] = dimId;

                                }else if((fpos + fcount) > (tupleOffset + tupleNum)){
                                        int tcount = tupleOffset + tupleNum - fpos ;
                                        for(int k=0;k<tcount;k++)
                                                factFilter[fpos+k-tupleOffset] = dimId;
                                }else{
                                        for(int k=0;k<fcount;k++)
                                                factFilter[fpos+k-tupleOffset] = dimId;

                                }

                                break;
                        }
                }
        }

}

__kernel  void count_join_result(__global int* num, __global int* psum, __global char* bucket, __global char* fact, long inNum, __global int* count, __global int * factFilter){
        int lcount = 0;
	size_t stride = get_global_size(0);
	size_t offset = get_global_id(0);

        for(size_t i=offset;i<inNum;i+=stride){
                int fkey = ((int *)(fact))[i];
                int hkey = fkey &(HSIZE-1);
                int keyNum = num[hkey];

                for(int j=0;j<keyNum;j++){
                        int pSum = psum[hkey];
                        int dimKey = ((int *)(bucket))[2*j + 2*pSum];
                        int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];
                        if( dimKey == fkey){
                                lcount ++;
                                factFilter[i] = dimId;
                                break;
                        }
                }
        }

        count[offset] = lcount;
}

__kernel void rle_psum(__global int *count, __global char * fact,  long  tupleNum, long tupleOffset, __global int * filter){

	size_t offset = get_global_id(0);
	size_t stride = get_global_size(0);

        struct rleHeader *rheader = (struct rleHeader *) fact;
        int dNum = rheader->dictNum;

        for(size_t i= offset; i<dNum; i+= stride){

                int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
                int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];
                int lcount= 0;

                if((fcount + fpos) < tupleOffset)
                        continue;

                if(fpos >= (tupleOffset + tupleNum))
                        break;

                if(fpos < tupleOffset){
                        int tcount = fcount + fpos - tupleOffset;
                        if(tcount > tupleNum)
                                tcount = tupleNum;
                        for(int k=0;k<tcount;k++){
                                if(filter[k]!=0)
                                        lcount++;
                        }
                        count[i] = lcount;

                }else if ((fpos + fcount) > (tupleOffset + tupleNum)){
                        int tcount = tupleNum  + tupleOffset - fpos;
                        for(int k=0;k<tcount;k++){
                                if(filter[fpos-tupleOffset + k]!=0)
                                        lcount++;
                        }
                        count[i] = lcount;

                }else{
                        for(int k=0;k<fcount;k++){
                                if(filter[fpos-tupleOffset + k]!=0)
                                        lcount++;
                        }
                        count[i] = lcount;
                }
        }

}

__kernel void joinFact_rle(__global int *resPsum, __global char * fact,  int attrSize, long  tupleNum, long tupleOffset, __global int * filter, __global char * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);

        struct rleHeader *rheader = (struct rleHeader *) fact;
        int dNum = rheader->dictNum;

        for(size_t i = startIndex; i<dNum; i += stride){
                int fkey = ((int *)(fact+sizeof(struct rleHeader)))[i];
                int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
                int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

                if((fcount + fpos) < tupleOffset)
                        continue;

                if(fpos >= (tupleOffset + tupleNum))
                        break;

                if(fpos < tupleOffset){
                        int tcount = fcount + fpos - tupleOffset;
                        int toffset = resPsum[i];
                        for(int j=0;j<tcount;j++){
                                if(filter[j] != 0){
                                        ((int*)result)[toffset] = fkey ;
                                        toffset ++;
                                }
                        }

                }else if ((fpos + fcount) > (tupleOffset + tupleNum)){
                        int tcount = tupleOffset + tupleNum - fpos;
                        int toffset = resPsum[i];
                        for(int j=0;j<tcount;j++){
                                if(filter[fpos-tupleOffset+j] !=0){
                                        ((int*)result)[toffset] = fkey ;
                                        toffset ++;
                                }
                        }

                }else{
                        int toffset = resPsum[i];
                        for(int j=0;j<fcount;j++){
                                if(filter[fpos-tupleOffset+j] !=0){
                                        ((int*)result)[toffset] = fkey ;
                                        toffset ++;
                                }
                        }
                }
        }

}

__kernel void joinFact_dict_other(__global int *resPsum, __global char * fact,  __global char *dict, int byteNum,int attrSize, long  num, __global int * filter, __global char * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localOffset = resPsum[startIndex] * attrSize;

        struct dictHeader *dheader = (struct dictHeader*)dict;

        for(size_t i=startIndex;i<num;i+=stride){
                if(filter[i] != 0){
                        int key = 0;
			char *buf = (char *) &key;
			for(int k=0;k<byteNum;k++)
				buf[k] = (fact + i*byteNum)[k];
			buf = (char *)&dheader->hash[key];
			for(int k=0;k<attrSize;k++)
				(result + localOffset)[k] = buf[k];
                        localOffset += attrSize;
                }
        }
}

__kernel void joinFact_dict_int(__global int *resPsum, __global char * fact, __global char *dict, int byteNum, int attrSize, long  num, __global int * filter, __global char * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localCount = resPsum[startIndex];

        struct dictHeader *dheader = (struct dictHeader*)dict;

        for(size_t i=startIndex;i<num;i+=stride){
                if(filter[i] != 0){
                        int key = 0;
			char *buf = (char *)&key;
			for(int k=0;k<byteNum;k++)
				buf[k] = (fact + i *byteNum)[k];
                        ((int*)result)[localCount] = dheader->hash[key];
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

__kernel void joinFact_int(__global int *resPsum, __global char * fact,  int attrSize, long  num, __global int * filter, __global char * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localCount = resPsum[startIndex];

        for(size_t i=startIndex;i<num;i+=stride){
                if(filter[i] != 0){
                        ((int*)result)[localCount] = ((int *)fact)[i];
                        localCount ++;
                }
        }
}

__kernel void joinDim_rle(__global int *resPsum, __global char * dim, int attrSize, long tupleNum, long tupleOffset, __global int * filter, __global char * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localCount = resPsum[startIndex];

        struct rleHeader *rheader = (struct rleHeader *) dim;
        int dNum = rheader->dictNum;

        for(size_t i = startIndex; i<tupleNum; i += stride){
                int dimId = filter[i];
                if(dimId != 0){
                        for(int j=0;j<dNum;j++){
                                int dkey = ((int *)(dim+sizeof(struct rleHeader)))[j];
                                int dcount = ((int *)(dim+sizeof(struct rleHeader)))[j + dNum];
                                int dpos = ((int *)(dim+sizeof(struct rleHeader)))[j + 2*dNum];

                                if(dpos == dimId || ((dpos < dimId) && (dpos + dcount) > dimId)){
                                        ((int*)result)[localCount] = dkey ;
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

        struct dictHeader *dheader = (struct dictHeader*)dict;

        for(size_t i=startIndex;i<num;i+=stride){
                int dimId = filter[i];
                if( dimId != 0){
                        int key = 0;
			char *buf = (char *)&key;
			for(int k=0;k<byteNum;k++)
				buf[k] = (dim + (dimId-1) * byteNum)[k];
			buf = (char *)&dheader->hash[key];
			for(int k=0;k<attrSize;k++)
				(result+localOffset)[k] = buf[k];
                        localOffset += attrSize;
                }
        }
}

__kernel void joinDim_dict_int(__global int *resPsum, __global char * dim, __global char *dict, int byteNum, int attrSize, long num, __global int * filter, __global char * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localCount = resPsum[startIndex];

        struct dictHeader *dheader = (struct dictHeader*)dict;

        for(size_t i=startIndex;i<num;i+=stride){
                int dimId = filter[i];
                if( dimId != 0){
                        int key = 0;
			char * buf = (char *)&key;
			for(int k=0;k<byteNum;k++)
				buf[k] = (dim + (dimId-1)*byteNum)[k];
                        ((int*)result)[localCount] = dheader->hash[key];
                        localCount ++;
                }
        }
}

__kernel void joinDim_int(__global int *resPsum, __global char * dim, int attrSize, long num, __global int * filter, __global char * result){

	size_t startIndex = get_global_id(0);
	size_t stride = get_global_size(0);
        long localCount = resPsum[startIndex];

        for(size_t i=startIndex;i<num;i+=stride){
                int dimId = filter[i];
                if( dimId != 0){
                        ((int*)result)[localCount] = ((int*)dim)[dimId-1];
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

