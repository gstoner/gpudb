/*
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <cuda.h>
#include "../include/common.h"
#include "../include/gpuCudaLib.h"

#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
    }} while(0)

#define SAMPLE_STRIDE 128
#define SHARED_SIZE_LIMIT 1024 

__device__ static int gpu_strcmp(const char *s1, const char *s2, int len){
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
static inline __device__ int nextPowerOfTwo(int x)
{
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

static inline __host__ __device__ int iDivUp(int a, int b)
{
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static inline __host__ __device__ int getSampleCount(int dividend)
{
    return iDivUp(dividend, SAMPLE_STRIDE);
}

static inline __device__ int binarySearchInInt(int val, int *data, int L, int stride, int sortDir)
{
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

static inline __device__ int binarySearchExInt(int val, int *data, int L, int stride, int sortDir)
{
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


static inline __device__ int binarySearchIn(char * val, char *data, int L, int stride, int sortDir, int keySize)
{
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

static inline __device__ int binarySearchEx(char * val, char *data, int L, int stride, int sortDir, int keySize)
{
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

__global__ void generateSampleRanksKernel(
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
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

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

static void generateSampleRanks(
        int *d_RanksA,
        int *d_RanksB,
        char *d_SrcKey,
        int keySize,
        int stride,
        int N,
        int sortDir
)
{
        int lastSegmentElements = N % (2 * stride);
        int threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

        generateSampleRanksKernel<<<iDivUp(threadCount, 256), 256>>>(d_RanksA, d_RanksB, d_SrcKey, keySize,stride, N, threadCount, sortDir);
}



__global__ void mergeRanksAndIndicesKernel(
    int *d_Limits,
    int *d_Ranks,
    int stride,
    int N,
    int threadCount
)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

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


static void mergeRanksAndIndices(
    int *d_LimitsA,
    int *d_LimitsB,
    int *d_RanksA,
    int *d_RanksB,
    int stride,
    int N
)
{
    int lastSegmentElements = N % (2 * stride);
    int         threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

    mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
        d_LimitsA,
        d_RanksA,
        stride,
        N,
        threadCount
    );

    mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
        d_LimitsB,
        d_RanksB,
        stride,
        N,
        threadCount
    );
}

inline __device__ void merge(
        char *dstKey,
        int *dstVal,
        char *srcAKey,
        int *srcAVal,
        char *srcBKey,
        int *srcBVal,
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

    if (threadIdx.x < lenA)
    {
        memcpy(keyA, srcAKey + threadIdx.x*keySize, keySize);
        valA = srcAVal[threadIdx.x];
        dstPosA = binarySearchEx(keyA, srcBKey, lenB, nPowTwoLenB, sortDir,keySize) + threadIdx.x;
    }

    if (threadIdx.x < lenB)
    {
        memcpy(keyB, srcBKey + threadIdx.x * keySize, keySize);
        valB = srcBVal[threadIdx.x];
        dstPosB = binarySearchIn(keyB, srcAKey, lenA, nPowTwoLenA, sortDir, keySize) + threadIdx.x;
    }

    __syncthreads();

    if (threadIdx.x < lenA)
    {
        memcpy(dstKey + dstPosA*keySize, keyA, keySize);
        dstVal[dstPosA] = valA;
    }

    if (threadIdx.x < lenB)
    {
        memcpy(dstKey + dstPosB * keySize, keyB, keySize);
        dstVal[dstPosB] = valB;
    }
}

__global__ void mergeElementaryIntervalsKernel(
        char *d_DstKey,
        int *d_DstVal,
        char *d_SrcKey,
        int *d_SrcVal,
        int *d_LimitsA,
        int *d_LimitsB,
        int stride,
        int N,
        int sortDir,
        int keySize
)
{
__shared__ char s_key[2 * SAMPLE_STRIDE*64];
    __shared__ int s_val[2 * SAMPLE_STRIDE];

    const int   intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
    const int segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
    d_SrcKey += segmentBase * keySize;
    d_SrcVal += segmentBase;
    d_DstKey += segmentBase * keySize;
    d_DstVal += segmentBase;

    //Set up threadblock-wide parameters
    __shared__ int startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

    if (threadIdx.x == 0)
    {
        int segmentElementsA = stride;
        int segmentElementsB = umin(stride, N - segmentBase - stride);
        int  segmentSamplesA = getSampleCount(segmentElementsA);
        int  segmentSamplesB = getSampleCount(segmentElementsB);
        int   segmentSamples = segmentSamplesA + segmentSamplesB;

        startSrcA    = d_LimitsA[blockIdx.x];
        startSrcB    = d_LimitsB[blockIdx.x];
        int endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA;
        int endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB;
        lenSrcA      = endSrcA - startSrcA;
        lenSrcB      = endSrcB - startSrcB;
        startDstA    = startSrcA + startSrcB;
        startDstB    = startDstA + lenSrcA;
    }
//Load main input data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        memcpy(s_key + threadIdx.x * keySize, d_SrcKey + (startSrcA + threadIdx.x)*keySize, keySize);
        s_val[threadIdx.x +             0] = d_SrcVal[0 + startSrcA + threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        memcpy(s_key + (threadIdx.x + SAMPLE_STRIDE)*keySize, d_SrcKey + (stride + startSrcB+threadIdx.x)*keySize,keySize);
        s_val[threadIdx.x + SAMPLE_STRIDE] = d_SrcVal[stride + startSrcB + threadIdx.x];
    }

    //Merge data in shared memory
    __syncthreads();
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

    //Store merged data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        memcpy(d_DstKey + (startDstA + threadIdx.x)*keySize, s_key + threadIdx.x * keySize, keySize);
        d_DstVal[startDstA + threadIdx.x] = s_val[threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        memcpy(d_DstKey + (startDstB + threadIdx.x)*keySize, s_key + (lenSrcA + threadIdx.x)*keySize, keySize);
        d_DstVal[startDstB + threadIdx.x] = s_val[lenSrcA + threadIdx.x];
    }
}

static void mergeElementaryIntervals(
        char *d_DstKey,
        int *d_DstVal,
        char *d_SrcKey,
        int *d_SrcVal,
        int *d_LimitsA,
        int *d_LimitsB,
        int stride,
        int N,
        int sortDir,
        int keySize
)
{
        int lastSegmentElements = N % (2 * stride);
        int  mergePairs = (lastSegmentElements > stride) ? getSampleCount(N) : (N - lastSegmentElements) / SAMPLE_STRIDE;

        mergeElementaryIntervalsKernel<<<mergePairs, SAMPLE_STRIDE>>>(
            d_DstKey,
            d_DstVal,
            d_SrcKey,
            d_SrcVal,
            d_LimitsA,
            d_LimitsB,
            stride,
            N,
                sortDir,
                keySize
        );
}


static int *d_RanksA, *d_RanksB, *d_LimitsA, *d_LimitsB;
static const int MAX_SAMPLE_COUNT = 32768;

void initMergeSort(void)
{
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_RanksA,  MAX_SAMPLE_COUNT * sizeof(int)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_RanksB,  MAX_SAMPLE_COUNT * sizeof(int)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_LimitsA, MAX_SAMPLE_COUNT * sizeof(int)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_LimitsB, MAX_SAMPLE_COUNT * sizeof(int)));
}

void finishMergeSort(void){
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_RanksA));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_RanksB));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_LimitsA));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_LimitsB));
}

__device__ static inline void Comparator(
    char * keyA,
    int &valA,
    char * keyB,
    int &valB,
    int keySize,
    int dir
)
{
        int t;
        char buf[32];

    if ((gpu_strcmp(keyA,keyB,keySize) == 1) == dir)
    {
        memcpy(buf, keyA, keySize);
        memcpy(keyA, keyB, keySize);
        memcpy(keyB, buf, keySize);
        t = valA;
        valA = valB;
        valB = t;
    }
}

#define NTHREAD  (SHARED_SIZE_LIMIT/2)

__global__ static void sort_key(char * key, int tupleNum, int keySize, char *result, char *pos,int dir){
        int lid = threadIdx.x;
        int bid = blockIdx.x;

        __shared__ char bufKey[SHARED_SIZE_LIMIT * 32];
        __shared__ int bufVal[SHARED_SIZE_LIMIT];

        int gid = bid * SHARED_SIZE_LIMIT + lid;

        memcpy(bufKey + lid*keySize, key + gid*keySize, keySize);
        bufVal[lid] = gid;
        memcpy(bufKey + (lid+blockDim.x)*keySize, key +(gid+blockDim.x)*keySize, keySize);
        bufVal[lid+blockDim.x] = gid+ blockDim.x;

        __syncthreads();

        for (int size = 2; size < tupleNum && size < SHARED_SIZE_LIMIT; size <<= 1){
                int ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

                for (int stride = size / 2; stride > 0; stride >>= 1){
                        __syncthreads();
                        int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
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
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                bufKey+pos*keySize, bufVal[pos +      0],
                bufKey+(pos+stride)*keySize, bufVal[pos + stride],
                keySize,
                dir
            );
        }
    }

    __syncthreads();

        memcpy(result + gid*keySize, bufKey + lid*keySize, keySize);

        ((int *)pos)[gid] = bufVal[lid];
        memcpy(result + (gid+blockDim.x)*keySize, bufKey + (lid+blockDim.x)*keySize,keySize);
        ((int *)pos)[gid+blockDim.x] = bufVal[lid+blockDim.x];

}

__global__ static void set_key(char *key, int tupleNum){

        int stride = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i=tid;i<tupleNum;i+=stride)
		key[i] = '{';

}

__global__ static void gather_result(char * keyPos, char ** col, int newNum, int tupleNum, int *size, int colNum, char **result){
        int stride = blockDim.x * gridDim.x;
        int index = blockIdx.x * blockDim.x + threadIdx.x;

	for(int j=0;j<colNum;j++){
			for(int i=index;i<newNum;i+=stride){
					int pos = ((int *)keyPos)[i];
			if(pos<tupleNum)
				memcpy(result[j] + i*size[j], col[j] +pos*size[j], size[j]);
		}
        }
}

__global__ void build_orderby_keys(char ** content, int tupleNum, int odNum, int keySize,int *index, int * size, char *key){
	int stride = blockDim.x * gridDim.x;
        int offset = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i=offset;i<tupleNum;i+=stride){
		int pos = i* keySize;
		
		for(int j=0;j<odNum;j++){
			memcpy(key+pos,content[index[j]]+i*size[j],size[j]);
			pos += size[j];
		}

	}
}



//only handle uncompressed data
//if the data are compressed, uncompress first

struct tableNode * orderBy(struct orderByNode * odNode, struct statistic *pp){
	struct tableNode * res = NULL;

	res = (struct tableNode *)malloc(sizeof(struct tableNode));
	CHECK_POINTER(res);
	res->tupleNum = odNode->table->tupleNum;
	res->totalAttr = odNode->table->totalAttr;
	res->tupleSize = odNode->table->tupleSize;

	res->attrType = (int *) malloc(sizeof(int) * res->totalAttr);
	CHECK_POINTER(res->attrType);
	res->attrSize = (int *) malloc(sizeof(int) * res->totalAttr);
	CHECK_POINTER(res->attrSize);
	res->attrTotalSize = (int *) malloc(sizeof(int) * res->totalAttr);
	CHECK_POINTER(res->attrTotalSize);
	res->dataPos = (int *) malloc(sizeof(int) * res->totalAttr);
	CHECK_POINTER(res->dataPos);
	res->dataFormat = (int *) malloc(sizeof(int) * res->totalAttr);
	CHECK_POINTER(res->dataFormat);
	res->content = (char **) malloc(sizeof(char *) * res->totalAttr);
	CHECK_POINTER(res->content);

	initMergeSort();

	int gpuTupleNum = odNode->table->tupleNum;
	char * gpuKey, **column, ** gpuContent;
	char * gpuSortedKey;
	int *gpuIndex,  *gpuSize;

	column = (char**) malloc(sizeof(char*) *res->totalAttr);
	CHECK_POINTER(column);
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuContent, sizeof(char *) * res->totalAttr));

	for(int i=0;i<res->totalAttr;i++){
		res->attrType[i] = odNode->table->attrType[i];
		res->attrSize[i] = odNode->table->attrSize[i];
		res->attrTotalSize[i] = odNode->table->attrTotalSize[i];
		res->dataPos[i] = MEM;
		res->dataFormat[i] = UNCOMPRESSED;
		res->content[i] = (char *) malloc( res->attrSize[i] * res->tupleNum);
		CHECK_POINTER(res->content[i]);

		int attrSize = res->attrSize[i];
		if(odNode->table->dataPos[i] == MEM){
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&column[i], attrSize *res->tupleNum));
			CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[i], odNode->table->content[i], attrSize*res->tupleNum, cudaMemcpyHostToDevice));
		}else if (odNode->table->dataPos[i] == GPU){
			column[i] = odNode->table->content[i];
		}

		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuContent[i], &column[i], sizeof(char *), cudaMemcpyHostToDevice));
	}

	int keySize = 0;
	int *cpuSize = (int *)malloc(sizeof(int) * odNode->orderByNum);
	CHECK_POINTER(cpuSize);

	for(int i=0;i<odNode->orderByNum;i++){
		int index = odNode->orderByIndex[i];
		cpuSize[i] = odNode->table->attrSize[index];
		keySize += odNode->table->attrSize[index];
	}

	int newNum = 1;

	while(newNum<gpuTupleNum){
		newNum *=2;
	}

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuSize, sizeof(int)* res->totalAttr));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuSize,cpuSize, sizeof(int)*odNode->orderByNum, cudaMemcpyHostToDevice));
	
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuKey, keySize * newNum));
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuSortedKey, keySize * newNum));

	set_key<<<512,128>>>(gpuKey,newNum*keySize);


	dim3 grid(512);
	dim3 block(NTHREAD);

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuIndex,res->totalAttr * sizeof(int)));
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuIndex, odNode->orderByIndex, sizeof(int) * odNode->orderByNum, cudaMemcpyHostToDevice));
	
	build_orderby_keys<<<grid,block>>>(gpuContent, gpuTupleNum, odNode->orderByNum, keySize,gpuIndex, gpuSize, gpuKey);

	char * gpuPos;

	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuPos, sizeof(int)*newNum));

	if(newNum < SHARED_SIZE_LIMIT){
		sort_key<<<1, newNum/2>>>(gpuKey, newNum, keySize,gpuSortedKey,(char *)gpuPos, 1);
	}else{
		int stageCount = 0;

			for (int i = SHARED_SIZE_LIMIT; i < newNum; i <<= 1, stageCount++);

			char *ikey, *okey;
			char *ival, *oval;
		char * d_BufKey, * d_BufVal;

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_BufKey, keySize * newNum));
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_BufVal, sizeof(int) * newNum));

		if (stageCount & 1){
			ikey = d_BufKey;
			ival = d_BufVal;
			okey = gpuSortedKey;
			oval = gpuPos;
		}else{
			ikey = gpuSortedKey;
			ival = gpuPos;
			okey = d_BufKey;
			oval = d_BufVal;
		}

		grid = newNum/NTHREAD;

		sort_key<<<grid, block>>>(gpuKey, newNum, keySize,gpuSortedKey,(char *)gpuPos, 1);

		for(int i=SHARED_SIZE_LIMIT;i<newNum;i*=2){
			int lastSegmentElements = newNum % (2 * i);

			generateSampleRanks(d_RanksA, d_RanksB, ikey,keySize, i, newNum, 1);

			CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());
			mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, i, newNum);

			mergeElementaryIntervals(okey, (int *)oval, ikey, (int *)ival, d_LimitsA, d_LimitsB, i, newNum, 1, keySize);

			if (lastSegmentElements <= i){
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(okey + (newNum - lastSegmentElements)*keySize, ikey + (newNum - lastSegmentElements)*keySize, lastSegmentElements * keySize, cudaMemcpyDeviceToDevice));
				CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(oval + (newNum - lastSegmentElements)*keySize, ival + (newNum - lastSegmentElements)*keySize, lastSegmentElements * keySize, cudaMemcpyDeviceToDevice));
			}

			char *t;
			t = ikey;
			ikey = okey;
			okey = t;
			t = ival;
			ival = oval;
			oval = t;
			}
	}	

	CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());
	char ** gpuResult;
	char ** result;
	
	result = (char**)malloc(sizeof(char *) * res->totalAttr);
	CHECK_POINTER(result);
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuResult, sizeof(char*)*res->totalAttr));
	for(int i=0;i<res->totalAttr;i++){
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&result[i], res->attrSize[i]* gpuTupleNum));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuResult[i], &result[i], sizeof(char*), cudaMemcpyHostToDevice));
	}

	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuSize, res->attrSize, sizeof(int) * res->totalAttr, cudaMemcpyHostToDevice););

        gather_result<<<512,64>>>(gpuPos, gpuContent, newNum, gpuTupleNum, gpuSize,res->totalAttr,gpuResult);

	for(int i=0; i<res->totalAttr;i++){
		int size = res->attrSize[i] * gpuTupleNum;
		memset(res->content[i],0, size);
		CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res->content[i], result[i],size, cudaMemcpyDeviceToHost));
	}

	for(int i=0;i<res->totalAttr;i++){
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[i]));
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(result[i]));
	}
	free(column);
	free(result);

	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuKey));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuContent));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuResult));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuIndex));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuSize));
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuPos));

	finishMergeSort();

	return res;
}
