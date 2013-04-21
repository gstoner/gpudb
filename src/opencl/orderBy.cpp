#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include <time.h>
#include "../include/common.h"
#include "../include/gpuOpenclLib.h"

#define SAMPLE_STRIDE 128
#define SHARED_SIZE_LIMIT 1024 
#define NTHREAD  (SHARED_SIZE_LIMIT/2)

static inline int iDivUp(int a, int b)
{
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static inline int getSampleCount(int dividend)
{
    return iDivUp(dividend, SAMPLE_STRIDE);
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
	int lastSegmentElements = N % (2 * stride);
    	int threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

	size_t globalSize = iDivUp(threadCount,256);
	size_t localSize = 256;


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
	d_LimitsA = clCreateBuffer(context->context,CL_MEM_READ_WRITE,sizeof(int)*MAX_SAMPLE_COUNT,NULL,0);
	d_LimitsB = clCreateBuffer(context->context,CL_MEM_READ_WRITE,sizeof(int)*MAX_SAMPLE_COUNT,NULL,0);
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
	
	struct timespec start,end;
        clock_gettime(CLOCK_REALTIME,&start);

	cl_event ndrEvt;
	cl_ulong startTime,endTime;

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

	long totalSize = 0;
	long * cpuOffset = (long *)malloc(sizeof(long) * res->totalAttr);
	long offset = 0;

	for(int i=0;i<res->totalAttr;i++){

		cpuOffset[i] = offset;
		res->attrType[i] = odNode->table->attrType[i];
		res->attrSize[i] = odNode->table->attrSize[i];
		res->attrTotalSize[i] = odNode->table->attrTotalSize[i];
		res->dataPos[i] = MEM;
		res->dataFormat[i] = UNCOMPRESSED;

		int size = res->attrSize[i] * res->tupleNum;

		if(size %4 !=0){
			size += (4 - size %4);
		}

		offset += size;
		totalSize += size;
	}

	gpuContent = clCreateBuffer(context->context,CL_MEM_READ_ONLY, totalSize, NULL, 0);

	for(int i=0;i<res->totalAttr;i++){

		int size = res->attrSize[i] * res->tupleNum;

		if(odNode->table->dataPos[i] == MEM){
			error = clEnqueueWriteBuffer(context->queue, gpuContent, CL_TRUE, cpuOffset[i], size, odNode->table->content[i],0,0,&ndrEvt);

			clWaitForEvents(1, &ndrEvt);
			clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
			clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
			pp->pcie += 1e-6 * (endTime - startTime);
		}else if (odNode->table->dataPos[i] == GPU){
			error = clEnqueueCopyBuffer(context->queue,(cl_mem)odNode->table->content[i],gpuContent,0,cpuOffset[i],size,0,0,0);
		}

	}

	cl_mem gpuOffset = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(long)*res->totalAttr,NULL,0);
	error = clEnqueueWriteBuffer(context->queue, gpuOffset, CL_TRUE, 0, sizeof(long)*res->totalAttr, cpuOffset,0,0,&ndrEvt);

	clWaitForEvents(1, &ndrEvt);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
	pp->pcie += 1e-6 * (endTime - startTime);

	free(cpuOffset);

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
	error = clEnqueueWriteBuffer(context->queue, gpuSize, CL_TRUE, 0, sizeof(int) * odNode->orderByNum, cpuSize,0,0,&ndrEvt);

	clWaitForEvents(1, &ndrEvt);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
	pp->pcie += 1e-6 * (endTime - startTime);

	gpuKey = clCreateBuffer(context->context,CL_MEM_READ_WRITE, keySize * newNum, NULL, 0);
	gpuSortedKey = clCreateBuffer(context->context,CL_MEM_READ_WRITE, keySize * newNum, NULL, 0);

	context->kernel = clCreateKernel(context->program,"set_key",0);

	long tmp = keySize * newNum;
	error = clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void *)&gpuKey);
	error = clSetKernelArg(context->kernel,1,sizeof(long), (void *)&tmp);

	localSize = 128;
	globalSize = 512 * localSize;

	error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

	gpuIndex = clCreateBuffer(context->context,CL_MEM_READ_ONLY, res->totalAttr * sizeof(int), NULL,0);
	error = clEnqueueWriteBuffer(context->queue, gpuIndex, CL_TRUE, 0, odNode->orderByNum * sizeof(int), odNode->orderByIndex,0,0,&ndrEvt);

	clWaitForEvents(1, &ndrEvt);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
	pp->pcie += 1e-6 * (endTime - startTime);

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

	cl_mem gpuPos = clCreateBuffer(context->context, CL_MEM_READ_WRITE, sizeof(int)*newNum, NULL,0);

	if(newNum <= SHARED_SIZE_LIMIT){

		context->kernel = clCreateKernel(context->program,"sort_key",0);

		int sortDir = 1;

		clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuKey);
		clSetKernelArg(context->kernel,1,sizeof(int), (void*)&newNum);
		clSetKernelArg(context->kernel,2,sizeof(int), (void*)&keySize);
		clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&gpuSortedKey);
		clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&gpuPos);
		clSetKernelArg(context->kernel,5,sizeof(int), (void*)&sortDir);
		clSetKernelArg(context->kernel,6,SHARED_SIZE_LIMIT*24, NULL);
		clSetKernelArg(context->kernel,7,SHARED_SIZE_LIMIT*sizeof(int), NULL);

		localSize = newNum/2;
		globalSize = localSize;
		error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);


	}else{
		int stageCount = 0;

       		for (int i = SHARED_SIZE_LIMIT; i < newNum; i <<= 1, stageCount++);

		cl_mem ikey, okey;
		cl_mem ival, oval;
		cl_mem d_BufKey, d_BufVal;

		d_BufKey = clCreateBuffer(context->context, CL_MEM_READ_WRITE, keySize * newNum, NULL,0);
		d_BufVal = clCreateBuffer(context->context, CL_MEM_READ_WRITE, sizeof(int) * newNum, NULL,0);

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

		globalSize = newNum/NTHREAD * localSize;

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

	long * resOffset = (long *) malloc(sizeof(long) * res->totalAttr);
	offset = 0;
	totalSize = 0;
	for(int i=0; i<res->totalAttr;i++){
		int size = res->attrSize[i] * res->tupleNum;
		if(size %4 != 0){
			size += 4 - (size % 4);
		}

		resOffset[i] = offset;
		offset += size;
		totalSize += size;
	}

	cl_mem gpuResult = clCreateBuffer(context->context,CL_MEM_READ_WRITE, totalSize, NULL,0);
	clEnqueueWriteBuffer(context->queue, gpuSize, CL_TRUE, 0, sizeof(int)*res->totalAttr, res->attrSize,0,0,&ndrEvt);

	clWaitForEvents(1, &ndrEvt);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
	pp->pcie += 1e-6 * (endTime - startTime);
	
	cl_mem gpuResOffset = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(long)*res->totalAttr, NULL,0);
	clEnqueueWriteBuffer(context->queue, gpuResOffset, CL_TRUE, 0 ,sizeof(long)*res->totalAttr, resOffset, 0,0,&ndrEvt);

	clWaitForEvents(1, &ndrEvt);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
	clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
	pp->pcie += 1e-6 * (endTime - startTime);


	context->kernel = clCreateKernel(context->program,"gather_result",0);
	clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpuPos);
	clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void*)&gpuContent);
	clSetKernelArg(context->kernel,2,sizeof(int),(void*)&newNum);
	clSetKernelArg(context->kernel,3,sizeof(int),(void*)&gpuTupleNum);
	clSetKernelArg(context->kernel,4,sizeof(cl_mem),(void*)&gpuSize);
	clSetKernelArg(context->kernel,5,sizeof(int),(void*)&res->totalAttr);
	clSetKernelArg(context->kernel,6,sizeof(cl_mem),(void*)&gpuResult);
	clSetKernelArg(context->kernel,7,sizeof(cl_mem),(void*)&gpuOffset);
	clSetKernelArg(context->kernel,8,sizeof(cl_mem),(void*)&gpuResOffset);

	localSize = 128;
	globalSize = 512 * localSize;
	
	error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

	for(int i=0; i<res->totalAttr;i++){
		int size = res->attrSize[i] * gpuTupleNum;
		res->content[i] = (char *) malloc( size);
		memset(res->content[i],0, size);
		clEnqueueReadBuffer(context->queue,gpuResult, CL_TRUE, resOffset[i], size, res->content[i],0,0,&ndrEvt);

		clWaitForEvents(1, &ndrEvt);
		clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,0);
		clGetEventProfilingInfo(ndrEvt,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,0);
		pp->pcie += 1e-6 * (endTime - startTime);
	}


	free(resOffset);
	clFinish(context->queue);
	clReleaseMemObject(gpuKey);
	clReleaseMemObject(gpuContent);
	clReleaseMemObject(gpuResult);
	clReleaseMemObject(gpuIndex);
	clReleaseMemObject(gpuSize);
	clReleaseMemObject(gpuPos);
	clReleaseMemObject(gpuOffset);
	clReleaseMemObject(gpuResOffset);
	finishMergeSort();

	clock_gettime(CLOCK_REALTIME,&end);
        double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
        printf("OrderBy Time: %lf\n", timeE/(1000*1000));

	return res;
}
