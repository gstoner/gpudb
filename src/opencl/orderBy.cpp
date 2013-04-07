#include <stdio.h>
#include <CL/cl.h>
#include "../include/common.h"
#include "../include/gpuOpenclLib.h"

#define SAMPLE_STRIDE 128
#define SHARED_SIZE_LIMIT 1024 
#define NTHREAD  (SHARED_SIZE_LIMIT/2)

static inline int iDivUp(int a, int b)
{
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static void generateSampleRanks(
        cl_mem d_RanksA,
        cl_mem d_RanksB,
        cl_mem d_SrcKey,
        int keySize,
        int stride,
        int N,
        int sortDir,
	struct clContext * context
)
{
        int lastSegmentElements = N % (2 * stride);
        int threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);
	size_t globalSize = iDivUp(threadCount,256);
	size_t localSize = 256;

	context->kernel = clCreateKernel(context->program,"generateSampleRanksKernel",0);

	clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&d_RanksA);
	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&d_RanksB);
	clSetKernelArg(context->kernel,2,sizeof(cl_mem), (void*)&d_SrcKey);
	clSetKernelArg(context->kernel,3,sizeof(int), (void*)&keySize);
	clSetKernelArg(context->kernel,4,sizeof(int), (void*)&stride);
	clSetKernelArg(context->kernel,5,sizeof(int), (void*)&N);
	clSetKernelArg(context->kernel,6,sizeof(int), (void*)&threadCount);
	clSetKernelArg(context->kernel,7,sizeof(int), (void*)&sortDir);

	clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);
}

static void mergeRanksAndIndices(
    	cl_mem d_LimitsA,
    	cl_mem d_LimitsB,
    	cl_mem d_RanksA,
    	cl_mem d_RanksB,
    	int stride,
    	int N,
	struct clContext * context
)
{
	size_t globalSize = iDivUp(threadCount,256);
	size_t localSize = 256;

	int lastSegmentElements = N % (2 * stride);
    	int threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

	context->kernel = clCreateKernel(context->program,"mergeRanksAndIndicesKernel",0); 
	
	clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&d_LimitsA);
	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&d_RanksA);
	clSetKernelArg(context->kernel,2,sizeof(int), (void*)&stride);
	clSetKernelArg(context->kernel,3,sizeof(int), (void*)&N);
	clSetKernelArg(context->kernel,4,sizeof(int), (void*)&threadCount);

	clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

	context->kernel = clCreateKernel(context->program,"mergeRanksAndIndicesKernel",0); 

	clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&d_LimitsB);
	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&d_RanksB);
	clSetKernelArg(context->kernel,2,sizeof(int), (void*)&stride);
	clSetKernelArg(context->kernel,3,sizeof(int), (void*)&N);
	clSetKernelArg(context->kernel,4,sizeof(int), (void*)&threadCount);

	clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

}


static void mergeElementaryIntervals(
        cl_mem d_DstKey,
        cl_mem d_DstVal,
        cl_mem d_SrcKey,
        cl_mem d_SrcVal,
        cl_mem d_LimitsA,
        cl_mem d_LimitsB,
        int stride,
        int N,
        int sortDir,
        int keySize,
	struct clContext * context
)
{
        int lastSegmentElements = N % (2 * stride);
        int  mergePairs = (lastSegmentElements > stride) ? getSampleCount(N) : (N - lastSegmentElements) / SAMPLE_STRIDE;

	size_t globalSize = mergePairs;
	size_t localSize = SAMPLE_STRIDE;

	context->kernel = clCreateKernel(context->program,"mergeElementaryIntervalsKernel",0);

	clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&d_DstKey);
	clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void*)&d_DstVal);
	clSetKernelArg(context->kernel,2,sizeof(cl_mem),(void*)&d_SrcKey);
	clSetKernelArg(context->kernel,3,sizeof(cl_mem),(void*)&d_SrcVal);
	clSetKernelArg(context->kernel,4,sizeof(cl_mem),(void*)&d_LimitsA);
	clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void*)&d_LimitsB);
	clSetKernelArg(context->kernel,6,sizeof(int),(void*)&stride);
	clSetKernelArg(context->kernel,7,sizeof(int),(void*)&N);
	clSetKernelArg(context->kernel,8,sizeof(int),(void*)&sortDir);
	clSetKernelArg(context->kernel,9,sizeof(int),(void*)&keySize);

	clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);
}

cl_mem d_RanksA, d_RanksB, d_LimitsA, d_LimitsB;
static const int MAX_SAMPLE_COUNT = 32768;

void initMergeSort(struct clContext *context){

	d_RanksA = clCreateBuffer(context->context,CL_MEM_READ_WRITE,sizeof(int)*MAX_SAMPLE_COUNT,NULL,0);
	d_RanksB = clCreateBuffer(context->context,CL_MEM_READ_WRITE,sizeof(int)*MAX_SAMPLE_COUNT,NULL,0);
	d_LimitsA = clCreateBuffer(context->context,CL_MEM_READ_WRITE,sizeof(int)*MAX_SAMPLE,NULL,0);
	d_LimitsB = clCreateBuffer(context->context,CL_MEM_READ_WRITE,sizeof(int)*MAX_SAMPLE,NULL,0);
}

void finishMergeSort(){
	clReleaseMemObject(d_RanksA);
	clReleaseMemObject(d_RanksB);
	clReleaseMemObject(d_LimitsA);
	clReleaseMemObject(d_LimitsB);
}


//only handle uncompressed data
//if the data are compressed, uncompress first

struct tableNode * orderBy(struct orderByNode * odNode, struct clContext *context, struct statistic *pp){
	
	struct tableNode * res = NULL;
	size_t globalSize, localSize;

	res = (struct tableNode *)malloc(sizeof(struct tableNode));
	res->tupleNum = odNode->table->tupleNum;
	res->totalAttr = odNode->table->totalAttr;
	res->tupleSize = odNode->table->tupleSize;

	res->attrType = (int *) malloc(sizeof(int) * res->totalAttr);
	res->attrSize = (int *) malloc(sizeof(int) * res->totalAttr);
	res->attrTotalSize = (int *) malloc(sizeof(int) * res->totalAttr);
	res->dataPos = (int *) malloc(sizeof(int) * res->totalAttr);
	res->dataFormat = (int *) malloc(sizeof(int) * res->totalAttr);
	res->content = (char **) malloc(sizeof(char *) * res->totalAttr);

	initMergeSort(context);

	int gpuTupleNum = odNode->table->tupleNum;
	cl_mem gpuKey, gpuContent;
	cl_mem gpuSortedKey;
	cl_mem gpuIndex, gpuSize;
	cl_int error = 0;

	gpuContent = clCreateBuffer(context->context,CL_MEM_READ_ONLY, res->tupleNum * res->tupleSize, NULL, 0);

	long * cpuOffset = (long *)malloc(sizeof(long) * res->totalAttr);
	long offset = 0;

	for(int i=0;i<res->totalAttr;i++){

		cpuOffset[i] = offset;
		res->attrType[i] = odNode->table->attrType[i];
		res->attrTotalSize[i] = odNode->table->attrTotalSize[i];
		res->dataPos[i] = MEM;
		res->dataFormat[i] = UNCOMPRESSED;

		res->attrSize[i] = odNode->table->attrSize[i];
		int attrSize = res->attrSize[i];
		res->content[i] = (char *) malloc( attrSize * res->tupleNum);

		if(odNode->table->dataPos[i] == MEM){
			error = clEnqueueWriteBuffer(context->queue, gpuContent, CL_TRUE, offset, attrSize * res->tupleNum, odNode->table->content[i],0,0,0);
		}else if (odNode->table->dataPos[i] == GPU){
			error = clEnqueueCopyBuffer(context->queue,(cl_mem)odNode->table->content[i],gpuContent,0,offset,res->tupleNum * attrSize,0,0,0);
		}

		offset += res->attrSize[i] * res->tupleNum;
	}

	cl_mem gpuOffset = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(long)*res->totalAttr,0);
	error = clWriteBuffer(context->queue, gpuOffset, CL_TRUE, 0, sizeof(long)*res->totalAttr, cpuOffset,0,0,0);

	int keySize = 0;
	int *cpuSize = (int *)malloc(sizeof(int) * odNode->orderByNum);

	for(int i=0;i<odNode->orderByNum;i++){
		int index = odNode->orderByIndex[i];
		cpuSize[i] = odNode->table->attrSize[index];
		keySize += odNode->table->attrSize[index];
	}

	int newNum = 1;

	while(newNum<gpuTupleNum){
		newNum *=2;
	}

	gpuSize = clCreateBuffer(context->context,CL_MEM_READ_ONLY, res->totalAttr * sizeof(int), NULL, 0);
	error = clEnqueueWriteBuffer(context->queue, gpuSize, CL_TRUE, 0, sizeof(int) * odNode->orderByNum, cpuSize,0,0,0);

	gpukey = clCreateBuffer(context->context,CL_MEM_READ_ONLY, keySize * newNum, NULL, 0);
	gpuSortedkey = clCreateBuffer(context->context,CL_MEM_READ_WRITE, keySize * newNum, NULL, 0);

	context->kernel = clCreateKernel(context->program,"set_key",0);

	clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void *)&gpuKey);
	clSetKernelArg(context->kernel,0,sizeof(int), (void *)&keySize);

	localSize = 128;
	globalSize = 512 * localSize;

	error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);


	dim3 grid(512);
	dim3 block(NTHREAD);

	gpuIndex =clCreateBuffer(context->queue,CL_MEM_READ_ONLY, res->totalAttr * sizeof(int), NULL,0);
	error = clEnqueueWriteBuffer(context->queue, gpuIndex, CL_TRUE, 0, odNode->orderByNum * sizeof(int), cpuSize,0,0,0);

	context->kernel = clCreateKernel(context->program,"build_orderby_keys",0);

	clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void *)&gpuContent);
	clSetKernelArg(context->kernel,1,sizeof(int), (void *)&gpuTupleNum);
	clSetKernelArg(context->kernel,2,sizeof(int), (void *)&odNode->orderByNum);
	clSetKernelArg(context->kernel,3,sizeof(int), (void *)&keySize);
	clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void *)&gpuIndex);
	clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void *)&gpuSize);
	clSetKernelArg(context->kernel,6,sizeof(cl_mem), (void *)&gpuKey);
	clSetKernelArg(context->kernel,7,sizeof(cl_mem), (void *)&gpuOffset);

	localSize = NTHREAD;
	globalSize = 512 * localSize;
	error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

	cl_mem gpuPos = clCreateBuffer(context->queue, CL_MEM_READ_WRITE, sizeof(int)*newNum, NULL,0);

	if(newNum < SHARED_SIZE_LIMIT){

		context->kernel = clCreateKernel(context->program,"sort_key",0);

		int sortDir = 1;

		clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuKey);
		clSetKernelArg(context->kernel,1,sizeof(int), (void*)&newNum);
		clSetKernelArg(context->kernel,2,sizeof(int), (void*)&keySize);
		clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&gpuSortedKey);
		clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&gpuPos);
		clSetKernelArg(context->kernel,5,sizeof(int), (void*)&sortDir);

		localSize = newNum/2;
		globalSize = localSize;
		error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

	}else{
		int stageCount = 0;

       		for (int i = SHARED_SIZE_LIMIT; i < newNum; i <<= 1, stageCount++);

		cl_mem ikey, okey;
		cl_mem ival, oval;
		cl_mem d_BufKey, d_BufVal;

		d_BufKey = clCreateBuffer(context->queue, CL_MEM_READ_WRITE, keySize * newNum, NULL,0);
		d_BufVal = clCreateBuffer(context->queue, CL_MEM_READ_WRITE, sizeof(int) * newNum, NULL,0);

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

		glocalSize = newNum/NTHREAD * localSize;

		context->kernel = clCreateKernel(context->program,"sort_key",0);

		int sortDir = 1;
		clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuKey);
		clSetKernelArg(context->kernel,1,sizeof(int), (void*)&newNum);
		clSetKernelArg(context->kernel,2,sizeof(int), (void*)&keySize);
		clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&gpuSortedKey);
		clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&gpuPos);
		clSetKernelArg(context->kernel,5,sizeof(int), (void*)&sortDir);

		error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

		for(int i=SHARED_SIZE_LIMIT;i<newNum;i*=2){
			int lastSegmentElements = newNum % (2 * i);

			generateSampleRanks(d_RanksA, d_RanksB, ikey,keySize, i, newNum, 1, context);

			mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, i, newNum, context);

			mergeElementaryIntervals(okey, oval, ikey, ival, d_LimitsA, d_LimitsB, i, newNum, 1, keySize,context);

			if (lastSegmentElements <= i){
				clEnqueueCopyBuffer(context->queue, ikey, okey, (newNum-lastSegmentElements)*keySize,(newNum-lastSegmentElements)*keySize, lastSegmentElements*keySize, 0,0,0);
				clEnqueueCopyBuffer(context->queue, ival, oval, (newNum-lastSegmentElements)*keySize,(newNum-lastSegmentElements)*keySize, lastSegmentElements*keySize, 0,0,0);
			}

			cl_mem t;
			t = ikey;
			ikey = okey;
			okey = t;
			t = ival;
			ival = oval;
			oval = t;
        	}
	}	

	cl_mem gpuResult = clCreateBuffer(context->queue,CL_MEM_READ_WRITE, res->tupleNum * res->tupleSize, NULL,0);
	clEnqueueWriteBuffer(context->queue, gpuSize, CL_TRUE, 0, sizeof(int)*res->totalAttr, gb->table->content[i],0,0,0);

	context->kernel = clCreateKernel(context->program,"gather_result",0);
	clSetkernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpuPos);
	clSetkernelArg(context->kernel,1,sizeof(cl_mem),(void*)&gpuContent);
	clSetkernelArg(context->kernel,2,sizeof(int),(void*)&newNum);
	clSetkernelArg(context->kernel,3,sizeof(int),(void*)&gpuTupleNum);
	clSetkernelArg(context->kernel,4,sizeof(cl_mem),(void*)&gpuSize);
	clSetkernelArg(context->kernel,5,sizeof(int),(void*)&res->totalAttr);
	clSetkernelArg(context->kernel,6,sizeof(cl_mem),(void*)&gpuResult);

	localSize = 128;
	globalSize = 512 * localSize;
	
	error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

	int offset = 0;
	for(int i=0; i<res->totalAttr;i++){
		int size = res->attrSize[i] * gpuTupleNum;
		memset(res->content[i],0, size);
		clReadBuffer(context->queue,gpuResult, CL_TRUE, offset, size, res->content[i],0);
		offset += size;
	}


	clReleaseMemObject(gpuKey);
	clReleaseMemObject(gpuContent);
	clReleaseMemObject(gpuResult);
	clReleaseMemObject(gpuIndex);
	clReleaseMemObject(gpuSize);
	clReleaseMemObject(gpuPos);
	finishSortMerge();

	return res;
}
