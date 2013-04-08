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
	cl_mem gpuResult;
	cl_mem gpuAttrSize;

	long totalSize = tn->tupleNum * tn->tupleSize;

	cl_int error = 0;

	struct timespec start,end;

	clock_gettime(CLOCK_REALTIME,&start);

	cl_mem gpuContent = clCreateBuffer(context->context, CL_MEM_READ_ONLY, totalSize, NULL, &error);
	gpuResult = clCreateBuffer(context->context, CL_MEM_READ_WRITE, totalSize, NULL, &error);
	gpuAttrSize = clCreateBuffer(context->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int)*tn->totalAttr,tn->attrSize,&error);

	res = (char *) malloc(totalSize);

	long offset = 0;
	long *colOffset = (long*)malloc(sizeof(long)*tn->totalAttr);

	for(int i=0;i<tn->totalAttr;i++){
		colOffset[i] = offset;
		int size = tn->tupleNum * tn->attrSize[i]; 

		if(tn->dataPos[i] == MEM)
			clEnqueueWriteBuffer(context->queue,gpuContent,CL_TRUE,offset,size,tn->content[i],0,0,0);
		else
			clEnqueueCopyBuffer(context->queue,(cl_mem)tn->content[i],gpuContent,0,offset,size,0,0,0);
			
		offset += size;
	}

	cl_mem gpuColOffset = clCreateBuffer(context->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(long)*tn->totalAttr,colOffset,&error);

	size_t globalSize = 512;
	size_t localSize = 128;

	context->kernel = clCreateKernel(context->program,"materialize",0);
	clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuContent);
	clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&gpuColOffset);
	clSetKernelArg(context->kernel,2,sizeof(int), (void*)&tn->totalAttr);
	clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&gpuAttrSize);
	clSetKernelArg(context->kernel,4,sizeof(long), (void*)&tn->tupleNum);
	clSetKernelArg(context->kernel,5,sizeof(int), (void*)&tn->tupleSize);
	clSetKernelArg(context->kernel,6,sizeof(cl_mem), (void*)&gpuResult);

	clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

	clEnqueueWriteBuffer(context->queue,gpuResult,CL_TRUE,0,totalSize,res,0,0,0);

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
