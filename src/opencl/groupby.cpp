#include <stdio.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>
#include "../include/common.h"
#include "../include/gpuOpenclLib.h"
#include "scanImpl.cpp"


/* 
 * groupBy: group by the data and calculate. 
 * 
 * Prerequisite:
 *	input data are not compressed
 *
 * Input:
 *	gb: the groupby node which contains the input data and groupby information
 *	pp: records the statistics such as kernel execution time 
 *
 * Return:
 *	a new table node
 */


struct tableNode * groupBy(struct groupByNode * gb, struct clContext * context, struct statistic * pp){

	struct tableNode * res = NULL;
	long gpuTupleNum;
	int gpuGbColNum;
	cl_mem gpuGbIndex;
	cl_mem gpuGbType, gpuGbSize;

	cl_mem gpuGbKey;
	cl_mem gpuContent;

	int gbCount;				// the number of groups
	int gbConstant = 0;			// whether group by constant

	cl_int error = 0;

	res = (struct tableNode *) malloc(sizeof(struct tableNode));
	res->tupleSize = gb->tupleSize;
	res->totalAttr = gb->outputAttrNum;
	res->attrType = (int *) malloc(sizeof(int) * res->totalAttr);
	res->attrSize = (int *) malloc(sizeof(int) * res->totalAttr);
	res->attrTotalSize = (int *) malloc(sizeof(int) * res->totalAttr);
	res->dataPos = (int *) malloc(sizeof(int) * res->totalAttr);
	res->dataFormat = (int *) malloc(sizeof(int) * res->totalAttr);
	res->content = (char **) malloc(sizeof(char **) * res->totalAttr);

	for(int i=0;i<res->totalAttr;i++){
		res->attrType[i] = gb->attrType[i];
		res->attrSize[i] = gb->attrSize[i];
		res->dataFormat[i] = UNCOMPRESSED;
	}
	
	gpuTupleNum = gb->table->tupleNum;
	gpuGbColNum = gb->groupByColNum;

	if(gpuGbColNum == 1 && gb->groupByIndex[0] == -1){

		gbConstant = 1;
	}

	size_t globalSize = 1024;
	size_t localSize = 128;

	cl_mem gpu_hashNum;
	cl_mem gpu_psum;
	cl_mem gpuGbCount;

	gpuContent = clCreateBuffer(context->context,CL_MEM_READ_ONLY, gb->table->tupleSize * gb->table->tupleNum,NULL,&error);

	long * cpuOffset = (long *)malloc(sizeof(long) * gb->table->totalAttr);
	long offset = 0;

	for(int i=0;i<gb->table->totalAttr;i++){
		int attrSize = gb->table->attrSize[i];
		cpuOffset[i] = offset;
		if(gb->table->dataPos[i]==MEM)
			error = clEnqueueWriteBuffer(context->queue, gpuContent, CL_TRUE, offset, attrSize * gb->table->tupleNum, gb->table->content[i],0,0,0);
		else
			error = clEnqueueCopyBuffer(context->queue,(cl_mem)gb->table->content[i],gpuContent,0,offset,gb->table->tupleNum * attrSize,0,0,0);

		offset += attrSize * gb->table->tupleNum;
	}

	cl_mem gpuOffset = clCreateBuffer(context->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(long)*gb->table->totalAttr,cpuOffset,&error);

	if(gbConstant != 1){

		gpuGbType = clCreateBuffer(context->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(int)*gb->groupByColNum,gb->groupByType,&error);
		gpuGbSize = clCreateBuffer(context->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(int)*gb->groupByColNum,gb->groupBySize,&error);
		gpuGbKey = clCreateBuffer(context->context,CL_MEM_READ_WRITE,sizeof(int)*gb->table->tupleNum,NULL,&error);

		gpuGbIndex = clCreateBuffer(context->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int)*gb->groupByColNum,gb->groupByIndex,&error);

		gpu_hashNum = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(int)*HSIZE,NULL,&error);

		context->kernel = clCreateKernel(context->program, "build_groupby_key",0);
		clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void *)&gpuContent);
		clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void *)&gpuOffset);
		clSetKernelArg(context->kernel,2,sizeof(int),(void *)&gpuGbColNum);
		clSetKernelArg(context->kernel,3,sizeof(cl_mem),(void *)&gpuGbIndex);
		clSetKernelArg(context->kernel,4,sizeof(cl_mem),(void *)&gpuGbType);
		clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void *)&gpuGbSize);
		clSetKernelArg(context->kernel,6,sizeof(long),(void *)&gpuTupleNum);
		clSetKernelArg(context->kernel,7,sizeof(cl_mem),(void *)&gpuGbKey);
		clSetKernelArg(context->kernel,8,sizeof(cl_mem),(void *)&gpu_hashNum);

		error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

		clReleaseMemObject(gpuGbType);
		clReleaseMemObject(gpuGbSize);
		clReleaseMemObject(gpuGbIndex);

		gbCount = 1;

		int tmp = 0;
		gpuGbCount = clCreateBuffer(context->context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int),&tmp,&error);

		int hsize = HSIZE;
		context->kernel = clCreateKernel(context->program, "count_group_num",0);
		clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void *)&gpu_hashNum);
		clSetKernelArg(context->kernel,1,sizeof(int),(void *)&hsize);
		clSetKernelArg(context->kernel,2,sizeof(cl_mem),(void *)&gpuGbCount);
		error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

		clEnqueueReadBuffer(context->queue, gpuGbCount, CL_TRUE, 0, sizeof(int), &gbCount,0,0,0);

		gpu_psum = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(int)*HSIZE,NULL,&error);

		scanImpl(gpu_hashNum,HSIZE,gpu_psum,context,pp);

		clReleaseMemObject(gpuGbCount);
		clReleaseMemObject(gpu_hashNum);
	}

	if(gbConstant == 1)
		res->tupleNum = 1;
	else
		res->tupleNum = gbCount;

	printf("groupBy num %d\n",res->tupleNum);

	cl_mem gpuResult = clCreateBuffer(context->context,CL_MEM_READ_WRITE, res->tupleSize * res->tupleNum, NULL, &error);
	
	gpuGbType = clCreateBuffer(context->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int)*res->totalAttr, res->attrType, &error);
	gpuGbSize = clCreateBuffer(context->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int)*res->totalAttr, res->attrSize, &error);

	cl_mem gpuGbExp = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(struct mathExp)*res->totalAttr, NULL, &error);

	cl_mem mathexp = clCreateBuffer(context->context, CL_MEM_READ_ONLY, 2*sizeof(struct mathExp)*res->totalAttr,NULL, &error);

	struct mathExp tmpExp[2];
	int * cpuFunc = (int *) malloc(sizeof(int) * res->totalAttr);

	offset = 0;
	for(int i=0;i<res->totalAttr;i++){

		error = clEnqueueWriteBuffer(context->queue, gpuGbExp, CL_TRUE, offset, sizeof(struct mathExp), &(gb->gbExp[i].exp),0,0,0);
		offset += sizeof(struct mathExp);
		cpuFunc[i] = gb->gbExp[i].func;
		if(gb->gbExp[i].exp.opNum == 2){
			struct mathExp * tmpMath = (struct mathExp *) gb->gbExp[i].exp.exp;
			tmpExp[0].op = tmpMath[0].op;
			tmpExp[0].opNum = tmpMath[0].opNum;
			tmpExp[0].opType = tmpMath[0].opType;
			tmpExp[0].opValue = tmpMath[0].opValue;

			tmpExp[1].op = tmpMath[1].op;
			tmpExp[1].opNum = tmpMath[1].opNum;
			tmpExp[1].opType = tmpMath[1].opType;
			tmpExp[1].opValue = tmpMath[1].opValue;
			clEnqueueWriteBuffer(context->queue, mathexp, CL_TRUE, 2*i*sizeof(struct mathExp),2*sizeof(struct mathExp),tmpExp,0,0,0);
		}
	}

	cl_mem gpuFunc = clCreateBuffer(context->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int)*res->totalAttr, cpuFunc, &error);

	long *resOffset = (long *)malloc(sizeof(long)*res->totalAttr);
	
	offset = 0;
	for(int i=0;i<res->totalAttr;i++){
		resOffset[i] = offset;
		offset += res->attrSize[i] * res->tupleNum;
	}

	cl_mem gpuResOffset = clCreateBuffer(context->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(long)*res->totalAttr, resOffset,&error);

	gpuGbColNum = res->totalAttr;

	if(gbConstant !=1){
		context->kernel = clCreateKernel(context->program,"agg_cal",0);
		clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuContent);
		clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&gpuOffset);
		clSetKernelArg(context->kernel,2,sizeof(int), (void*)&gpuGbColNum);
		clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&gpuGbExp);
		clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&mathexp);
		clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void*)&gpuGbType);
		clSetKernelArg(context->kernel,6,sizeof(cl_mem), (void*)&gpuGbSize);
		clSetKernelArg(context->kernel,7,sizeof(long), (void*)&gpuTupleNum);
		clSetKernelArg(context->kernel,8,sizeof(cl_mem), (void*)&gpuGbKey);
		clSetKernelArg(context->kernel,9,sizeof(cl_mem), (void*)&gpu_psum);
		clSetKernelArg(context->kernel,10,sizeof(cl_mem), (void*)&gpuResult);
		clSetKernelArg(context->kernel,11,sizeof(cl_mem), (void*)&gpuResOffset);
		clSetKernelArg(context->kernel,12,sizeof(cl_mem), (void*)&gpuFunc);

		error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);
		
		clReleaseMemObject(gpuGbKey);
		clReleaseMemObject(gpu_psum);
	}else{
		context->kernel = clCreateKernel(context->program,"agg_cal_cons",0);
		clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&gpuContent);
		clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void*)&gpuOffset);
		clSetKernelArg(context->kernel,2,sizeof(int), (void*)&gpuGbColNum);
		clSetKernelArg(context->kernel,3,sizeof(cl_mem), (void*)&gpuGbExp);
		clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&mathexp);
		clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void*)&gpuGbType);
		clSetKernelArg(context->kernel,6,sizeof(cl_mem), (void*)&gpuGbSize);
		clSetKernelArg(context->kernel,7,sizeof(long), (void*)&gpuTupleNum);
		clSetKernelArg(context->kernel,8,sizeof(cl_mem), (void*)&gpuResult);
		clSetKernelArg(context->kernel,9,sizeof(cl_mem), (void*)&gpuResOffset);
		clSetKernelArg(context->kernel,10,sizeof(cl_mem), (void*)&gpuFunc);

		error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);
	}

	for(int i=0; i<res->totalAttr;i++){
		res->content[i] = (char *)clCreateBuffer(context->context,CL_MEM_READ_WRITE, res->attrSize[i]*res->tupleNum, NULL, &error); 
		res->dataPos[i] = GPU;
		res->attrTotalSize[i] = res->tupleNum * res->attrSize[i];
		clEnqueueCopyBuffer(context->queue, gpuResult, (cl_mem)res->content[i], resOffset[i],0, res->attrSize[i] * res->tupleNum, 0,0,0);
	}

	free(resOffset);
	free(cpuOffset);

	clFinish(context->queue);
	clReleaseMemObject(gpuContent);
	clReleaseMemObject(gpuResult);
	clReleaseMemObject(gpuOffset);
	clReleaseMemObject(gpuResOffset);
	clReleaseMemObject(gpuGbExp);
	clReleaseMemObject(gpuFunc);

	return res;
}
