#include <stdio.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>
#include "../include/common.h"
#include "../include/schema.h"
#include "../include/gpuOpenclLib.h"

void * materializeCol(struct materializeNode * mn, struct clContext * context, struct statistic * pp){

	struct tableNode *tn = mn->table;
	char * res;
	cl_mem gpuResult, gpuContext;
	cl_mem gpuAttrSize;

	long size = tn->tupleNum * tn->tupleSize;

	cl_int error = 0;
	cl_kernel kernel;

	struct timespec start,end;

	clock_gettime(CLOCK_REALTIME,&start);

	cl_mem gpuContent = clCreateBuffer(context->context, CL_MEM_READ_ONLY, size, NULL, &error);
	gpuResult = clCreateBuffer(context->context, CL_MEM_READ_WRITE, size, NULL, &error);
	gpuAttrSize = clCreateBuffer(context->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int)*tn->totalAttr,tn->attrSize,&error);

	res = (char *) malloc(size);

	long offset = 0;
	long *colOffset = (long*)malloc(sizeof(long)*tn->totalAttr);
	for(int i=0;i<tn->totalAttr;i++){
		colOffset[i] = offset;
		clEnqueueWriteBuffer(context->queue,gpuContent,CL_TRUE,offset,tn->tupleNum * tn->attrSize[i],tn->content[i],0,0,0);
		offset += tn->tupleNum * tn->attrSize[i];
	}

	cl_mem gpuColOffset = clCreateBuffer(context->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(long)*tn->totalAttr,colOffset,&error);

	size_t globalSize = 512;
	size_t localSize = 128;

	size_t threadNum = globalSize;

	kernel = clCreateKernel(context->program,"materialize",0);
	clSetKernelArg(kernel,0,sizeof(cl_mem), (void*)&gpuContent);
	clSetKernelArg(kernel,1,sizeof(cl_mem), (void*)&gpuColOffset);
	clSetKernelArg(kernel,2,sizeof(int), (void*)&tn->totalAttr);
	clSetKernelArg(kernel,3,sizeof(cl_mem), (void*)&gpuAttrSize);
	clSetKernelArg(kernel,4,sizeof(long), (void*)&tn->tupleNum);
	clSetKernelArg(kernel,5,sizeof(int), (void*)&tn->tupleSize);
	clSetKernelArg(kernel,6,sizeof(cl_mem), (void*)&gpuResult);

	clEnqueueNDRangeKernel(context->queue, kernel, 1, 0, &threadNum,0,0,0,0);

	clEnqueueReadBuffer(context->queue,gpuResult,CL_TRUE,0,size,res,0,0,0);

	free(colOffset);

	clReleaseMemObject(gpuColOffset);
	clReleaseMemObject(gpuContent);
	clReleaseMemObject(gpuAttrSize);
	clReleaseMemObject(gpuResult);

	clock_gettime(CLOCK_REALTIME,&end);
	double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
	pp->total += timeE/(1000*1000) ;
	return res;
}
