#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <CL/cl.h>
#include "../include/common.h"
#include "../include/gpuOpenclLib.h"
#include "scanImpl.cpp"

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

struct tableNode * tableScan(struct scanNode *sn, struct clContext *context, struct statistic *pp){

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

	cl_int error = 0;

	cl_mem * column;

	long totalTupleNum = sn->tn->tupleNum;

	size_t localSize = 256;
	int blockNum = totalTupleNum / localSize + 1;

	if(blockNum >1024)
		blockNum = 1024;

	size_t globalSize = blockNum * localSize; 

	size_t threadNum = globalSize;
	int attrNum;

	attrNum = sn->whereAttrNum;
	column = (cl_mem *) malloc(attrNum * sizeof(cl_mem));

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

	long count = 0;

	cl_mem gpuFilter = clCreateBuffer(context->context, CL_MEM_READ_WRITE, sizeof(int)*totalTupleNum, NULL, &error);
	cl_mem gpuPsum = clCreateBuffer(context->context, CL_MEM_READ_WRITE, sizeof(int)*threadNum, NULL, &error);
	cl_mem gpuCount = clCreateBuffer(context->context, CL_MEM_READ_WRITE, sizeof(int)*threadNum, NULL, &error);

	assert(sn->hasWhere !=0);
	assert(sn->filter != NULL);

	struct whereCondition *where = sn->filter;

	if(1){

		cl_mem gpuExp = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(struct whereExp), NULL,&error);
		clEnqueueWriteBuffer(context->queue,gpuExp,CL_TRUE,0,sizeof(struct whereExp),&where->exp[0],0,0,0);

		int whereIndex = where->exp[0].index;
		int index = sn->whereIndex[whereIndex];
		int prevWhere = whereIndex;
		int prevIndex = index;

		int format = sn->tn->dataFormat[index];

		int prevFormat = format;
		int dNum;
		int byteNum;

		cl_mem gpuDictFilter;

		if(sn->tn->dataPos[index] == MEM)
			column[whereIndex] = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sn->tn->attrTotalSize[index], NULL, &error);
		else if(sn->tn->dataPos[index] == PINNED)
			column[whereIndex] = clCreateBuffer(context->context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, sn->tn->attrTotalSize[index], NULL, &error);

		if(format == UNCOMPRESSED){
			if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
				clEnqueueWriteBuffer(context->queue,column[whereIndex],CL_TRUE,0,sn->tn->attrTotalSize[index],sn->tn->content[index],0,0,0);
			else if (sn->tn->dataPos[index] == UVA)
				column[whereIndex] = (cl_mem) sn->tn->content[index];

			if(sn->tn->attrType[index] == INT){
				int rel = where->exp[0].relation;
				int whereValue = *((int*) where->exp[0].content);

				if(rel==EQ)
					context->kernel = clCreateKernel(context->program, "genScanFilter_init_int_eq", 0);
				else if(rel == GTH)
					context->kernel = clCreateKernel(context->program, "genScanFilter_init_int_gth", 0);
				else if(rel == LTH)
					context->kernel = clCreateKernel(context->program, "genScanFilter_init_int_lth", 0);
				else if(rel == GEQ)
					context->kernel = clCreateKernel(context->program, "genScanFilter_init_int_geq", 0);
				else if (rel == LEQ)
					context->kernel = clCreateKernel(context->program, "genScanFilter_init_int_leq", 0);

				clSetKernelArg(context->kernel, 0, sizeof(cl_mem), (void *)&column[whereIndex]);
				clSetKernelArg(context->kernel, 1, sizeof(long), (void *)&totalTupleNum);
				clSetKernelArg(context->kernel, 2, sizeof(int), (void *)&whereValue);
				clSetKernelArg(context->kernel, 3, sizeof(cl_mem), (void *)&gpuFilter);
				clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

			}else if (sn->tn->attrType[index] == FLOAT){
				int rel = where->exp[0].relation;
				float whereValue = *((int*) where->exp[0].content);

				if(rel==EQ)
					context->kernel = clCreateKernel(context->program, "genScanFilter_init_float_eq", 0);
				else if(rel == GTH)
					context->kernel = clCreateKernel(context->program, "genScanFilter_init_float_gth", 0);
				else if(rel == LTH)
					context->kernel = clCreateKernel(context->program, "genScanFilter_init_float_lth", 0);
				else if(rel == GEQ)
					context->kernel = clCreateKernel(context->program, "genScanFilter_init_float_geq", 0);
				else if (rel == LEQ)
					context->kernel = clCreateKernel(context->program, "genScanFilter_or_float_leq", 0);

				clSetKernelArg(context->kernel, 0, sizeof(cl_mem), (void *)&column[whereIndex]);
				clSetKernelArg(context->kernel, 1, sizeof(long), (void *)&totalTupleNum);
				clSetKernelArg(context->kernel, 2, sizeof(float), (void *)&whereValue);
				clSetKernelArg(context->kernel, 3, sizeof(cl_mem), (void *)&gpuFilter);
				clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

			}else{
				context->kernel = clCreateKernel(context->program, "genScanFilter_init", 0);
				clSetKernelArg(context->kernel, 0, sizeof(cl_mem), (void *)&column[whereIndex]);
				clSetKernelArg(context->kernel, 1, sizeof(int), (void *)&sn->tn->attrSize[index]);
				clSetKernelArg(context->kernel, 2, sizeof(int), (void *)&sn->tn->attrType[index]);
				clSetKernelArg(context->kernel, 3, sizeof(long), (void *)&totalTupleNum);
				clSetKernelArg(context->kernel, 4, sizeof(cl_mem), (void *)&gpuExp);
				clSetKernelArg(context->kernel, 5, sizeof(cl_mem), (void *)&gpuFilter);
				clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);
			}

		}else if(format == DICT){
			struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
			dNum = dheader->dictNum;
			byteNum = dheader->bitNum/8;

			cl_mem gpuDictHeader = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(struct dictHeader), NULL,&error);
			clEnqueueWriteBuffer(context->queue,gpuDictHeader,CL_TRUE,0,sizeof(struct dictHeader),dheader,0,0,0);

			if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
				clEnqueueWriteBuffer(context->queue,column[whereIndex],CL_TRUE,0,sn->tn->attrTotalSize[index]-sizeof(struct dictHeader),sn->tn->content[index] + sizeof(struct dictHeader),0,0,0);
			else if (sn->tn->dataPos[index] == UVA)
				column[whereIndex] = (cl_mem) (sn->tn->content[index]+sizeof(struct dictHeader));

			gpuDictFilter = clCreateBuffer(context->context,CL_MEM_READ_WRITE,dNum * sizeof(int),NULL,&error);

			context->kernel = clCreateKernel(context->program,"genScanFilter_dict_init",0); 
			clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void *)&gpuDictHeader);
			clSetKernelArg(context->kernel,1,sizeof(int), (void*)&sn->tn->attrSize[index]);
			clSetKernelArg(context->kernel,2,sizeof(int), (void*)&sn->tn->attrType[index]);
			clSetKernelArg(context->kernel,3,sizeof(int), (void*)&dNum);
			clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&gpuExp);
			clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void*)&gpuDictFilter);

			error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

			clReleaseMemObject(gpuDictHeader);

		}else if(format == RLE){

			if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
				clEnqueueWriteBuffer(context->queue,column[whereIndex],CL_TRUE,0,sn->tn->attrTotalSize[index],sn->tn->content[index],0,0,0);
				
			else if (sn->tn->dataPos[index] == UVA)
				column[whereIndex] = (cl_mem)sn->tn->content[index];

			long offset = 0;
			context->kernel = clCreateKernel(context->program,"genScanFilter_rle",0); 
			clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void *)&column[whereIndex]);
			clSetKernelArg(context->kernel,1,sizeof(int), (void *)&sn->tn->attrSize[index]);
			clSetKernelArg(context->kernel,2,sizeof(int), (void *)&sn->tn->attrType[index]);
			clSetKernelArg(context->kernel,3,sizeof(long), (void *)&totalTupleNum);
			clSetKernelArg(context->kernel,4,sizeof(long), (void *)&offset);
			clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void *)&gpuExp);
			clSetKernelArg(context->kernel,6,sizeof(int), (void *)&where->andOr);
			clSetKernelArg(context->kernel,7,sizeof(cl_mem), (void *)&gpuFilter);
			clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

		}

		int dictFilter = 0;
		int dictFinal = OR;
		int dictInit = 1;

		for(int i=1;i<where->expNum;i++){
			whereIndex = where->exp[i].index;
			index = sn->whereIndex[whereIndex];
			format = sn->tn->dataFormat[index];
			
			clEnqueueWriteBuffer(context->queue,gpuExp,CL_TRUE,0,sizeof(struct whereExp),&where->exp[i],0,0,0);

			if(prevIndex != index){
				if(prevFormat == DICT){
					if(dictInit == 1){
						context->kernel = clCreateKernel(context->program,"transform_dict_filter_init",0);
						dictInit = 0;
					}else if(dictFinal == OR)
						context->kernel = clCreateKernel(context->program,"transform_dict_filter_or",0);
					else
						context->kernel = clCreateKernel(context->program,"transform_dict_filter_and",0); 

					clSetKernelArg(context->kernel,0, sizeof(cl_mem), (void*)&gpuDictFilter);
					clSetKernelArg(context->kernel,1, sizeof(cl_mem), (void*)&column[prevWhere]);
					clSetKernelArg(context->kernel,2, sizeof(long), (void*)&totalTupleNum);
					clSetKernelArg(context->kernel,3, sizeof(int), (void*)&dNum);
					clSetKernelArg(context->kernel,4, sizeof(cl_mem), (void*)&gpuFilter);
					clSetKernelArg(context->kernel,5, sizeof(int), (void*)&byteNum);
					clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

					clReleaseMemObject(gpuDictFilter);
					dictFinal = where->andOr;
				}

				if(whereFree[prevWhere] == 1 && (sn->tn->dataPos[prevIndex] == MEM || sn->tn->dataPos[prevIndex] == PINNED))
					clReleaseMemObject(column[prevWhere]);

				if(sn->tn->dataPos[index] == MEM)
					column[whereIndex] = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sn->tn->attrTotalSize[index], NULL, &error);
				else if (sn->tn->dataPos[index] == PINNED)
					column[whereIndex] = clCreateBuffer(context->context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, sn->tn->attrTotalSize[index], NULL, &error);

				if(format == DICT){
					if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
						clEnqueueWriteBuffer(context->queue,column[whereIndex],CL_TRUE,0,sn->tn->attrTotalSize[index]-sizeof(struct dictHeader),sn->tn->content[index] + sizeof(struct dictHeader),0,0,0);
					else if (sn->tn->dataPos[index] == UVA)
						column[whereIndex] = (cl_mem)(sn->tn->content[index]+sizeof(struct dictHeader));

					struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
					dNum = dheader->dictNum;
					byteNum = dheader->bitNum/8;

					cl_mem gpuDictHeader = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(struct dictHeader), NULL,&error);
					clEnqueueWriteBuffer(context->queue,gpuDictHeader,CL_TRUE,0,sizeof(struct dictHeader),dheader,0,0,0);

					gpuDictFilter = clCreateBuffer(context->context,CL_MEM_READ_WRITE,dNum * sizeof(int),NULL,&error);

					context->kernel = clCreateKernel(context->program,"genScanFilter_dict_init",0); 
					clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void *)&gpuDictHeader);
					clSetKernelArg(context->kernel,1,sizeof(int), (void*)&sn->tn->attrSize[index]);
					clSetKernelArg(context->kernel,2,sizeof(int), (void*)&sn->tn->attrType[index]);
					clSetKernelArg(context->kernel,3,sizeof(int), (void*)&dNum);
					clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&gpuExp);
					clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void*)&gpuDictFilter);

					clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);
					dictFilter= -1;
					clReleaseMemObject(gpuDictHeader);

				}else{
					if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
						clEnqueueWriteBuffer(context->queue,column[whereIndex],CL_TRUE,0,sn->tn->attrTotalSize[index],sn->tn->content[index],0,0,0);
					else if (sn->tn->dataPos[index] == UVA)
						column[whereIndex] = (cl_mem)sn->tn->content[index];
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
							context->kernel = clCreateKernel(context->program, "genScanFilter_and_int_eq", 0);
						else if(rel == GTH)
							context->kernel = clCreateKernel(context->program, "genScanFilter_and_int_gth", 0);
						else if(rel == LTH)
							context->kernel = clCreateKernel(context->program, "genScanFilter_and_int_lth", 0);
						else if(rel == GEQ)
							context->kernel = clCreateKernel(context->program, "genScanFilter_and_int_geq", 0);
						else if (rel == LEQ)
							context->kernel = clCreateKernel(context->program, "genScanFilter_and_int_leq", 0);

						clSetKernelArg(context->kernel, 0, sizeof(cl_mem), (void *)&column[whereIndex]);
						clSetKernelArg(context->kernel, 1, sizeof(long), (void *)&totalTupleNum);
						clSetKernelArg(context->kernel, 2, sizeof(int), (void *)&whereValue);
						clSetKernelArg(context->kernel, 3, sizeof(cl_mem), (void *)&gpuFilter);
						clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

					}else{
						int rel = where->exp[i].relation;
						int whereValue = *((int*) where->exp[i].content);
						if(rel==EQ)
							context->kernel = clCreateKernel(context->program, "genScanFilter_or_int_eq", 0);
						else if(rel == GTH)
							context->kernel = clCreateKernel(context->program, "genScanFilter_or_int_gth", 0);
						else if(rel == LTH)
							context->kernel = clCreateKernel(context->program, "genScanFilter_or_int_lth", 0);
						else if(rel == GEQ)
							context->kernel = clCreateKernel(context->program, "genScanFilter_or_int_geq", 0);
						else if (rel == LEQ)
							context->kernel = clCreateKernel(context->program, "genScanFilter_or_int_leq", 0);

						clSetKernelArg(context->kernel, 0, sizeof(cl_mem), (void *)&column[whereIndex]);
						clSetKernelArg(context->kernel, 1, sizeof(long), (void *)&totalTupleNum);
						clSetKernelArg(context->kernel, 2, sizeof(int), (void *)&whereValue);
						clSetKernelArg(context->kernel, 3, sizeof(cl_mem), (void *)&gpuFilter);
						clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

					}

				} else if (sn->tn->attrType[index] == FLOAT){
					if(where->andOr == AND){
						int rel = where->exp[i].relation;
						float whereValue = *((int*) where->exp[i].content);
						if(rel==EQ)
							context->kernel = clCreateKernel(context->program, "genScanFilter_and_float_eq", 0);
						else if(rel == GTH)
							context->kernel = clCreateKernel(context->program, "genScanFilter_and_float_gth", 0);
						else if(rel == LTH)
							context->kernel = clCreateKernel(context->program, "genScanFilter_and_float_lth", 0);
						else if(rel == GEQ)
							context->kernel = clCreateKernel(context->program, "genScanFilter_and_float_geq", 0);
						else if (rel == LEQ)
							context->kernel = clCreateKernel(context->program, "genScanFilter_and_float_leq", 0);

						clSetKernelArg(context->kernel, 0, sizeof(cl_mem), (void *)&column[whereIndex]);
						clSetKernelArg(context->kernel, 1, sizeof(long), (void *)&totalTupleNum);
						clSetKernelArg(context->kernel, 2, sizeof(float), (void *)&whereValue);
						clSetKernelArg(context->kernel, 3, sizeof(cl_mem), (void *)&gpuFilter);
						clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

					}else{
						int rel = where->exp[i].relation;
						float whereValue = *((int*) where->exp[i].content);
						if(rel==EQ)
							context->kernel = clCreateKernel(context->program, "genScanFilter_or_float_eq", 0);
						else if(rel == GTH)
							context->kernel = clCreateKernel(context->program, "genScanFilter_or_float_gth", 0);
						else if(rel == LTH)
							context->kernel = clCreateKernel(context->program, "genScanFilter_or_float_lth", 0);
						else if(rel == GEQ)
							context->kernel = clCreateKernel(context->program, "genScanFilter_or_float_geq", 0);
						else if (rel == LEQ)
							context->kernel = clCreateKernel(context->program, "genScanFilter_or_float_leq", 0);

						clSetKernelArg(context->kernel, 0, sizeof(cl_mem), (void *)&column[whereIndex]);
						clSetKernelArg(context->kernel, 1, sizeof(long), (void *)&totalTupleNum);
						clSetKernelArg(context->kernel, 2, sizeof(float), (void *)&whereValue);
						clSetKernelArg(context->kernel, 3, sizeof(cl_mem), (void *)&gpuFilter);
						clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);
					}
				}else{
					if(where->andOr == AND)
						context->kernel = clCreateKernel(context->program, "genScanFilter_and", 0);
					else
						context->kernel = clCreateKernel(context->program, "genScanFilter_or", 0);

					clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void *)&column[whereIndex]);
					clSetKernelArg(context->kernel,1,sizeof(int),(void *)&sn->tn->attrSize[index]);
					clSetKernelArg(context->kernel,2,sizeof(int),(void *)&sn->tn->attrType[index]);
					clSetKernelArg(context->kernel,3,sizeof(long),(void *)&totalTupleNum);
					clSetKernelArg(context->kernel,4,sizeof(cl_mem),(void *)&gpuExp);
					clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void *)&gpuFilter);
					clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);
				}

			}else if(format == DICT){

				struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
				dNum = dheader->dictNum;
				byteNum = dheader->bitNum/8;

				cl_mem gpuDictHeader = clCreateBuffer(context->context,CL_MEM_READ_ONLY, sizeof(struct dictHeader), NULL,&error);
				clEnqueueWriteBuffer(context->queue,gpuDictHeader,CL_TRUE,0,sizeof(struct dictHeader),dheader,0,0,0);

				if(dictFilter != -1){
					if(where->andOr == AND)
						context->kernel = clCreateKernel(context->program,"genScanFilter_dict_and",0); 
					else
						context->kernel = clCreateKernel(context->program,"genScanFilter_dict_or",0); 

					clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void *)&gpuDictHeader);
					clSetKernelArg(context->kernel,1,sizeof(int), (void*)&sn->tn->attrSize[index]);
					clSetKernelArg(context->kernel,2,sizeof(int), (void*)&sn->tn->attrType[index]);
					clSetKernelArg(context->kernel,3,sizeof(int), (void*)&dNum);
					clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void*)&gpuExp);
					clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void*)&gpuDictFilter);

					clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);
				}

				dictFilter = 0;

				clReleaseMemObject(gpuDictHeader);

			}else if (format == RLE){
				//CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[index], sn->content[index], sn->whereSize[index], cudaMemcpyHostToDevice));
				context->kernel = clCreateKernel(context->program,"genScanFilter_rle",0); 

				long offset = 0;
				clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void *)&column[whereIndex]);
				clSetKernelArg(context->kernel,1,sizeof(int), (void *)&sn->tn->attrSize[index]);
				clSetKernelArg(context->kernel,2,sizeof(int), (void *)&sn->tn->attrType[index]);
				clSetKernelArg(context->kernel,3,sizeof(long), (void *)&totalTupleNum);
				clSetKernelArg(context->kernel,4,sizeof(long), (void *)&offset);
				clSetKernelArg(context->kernel,5,sizeof(cl_mem), (void *)&gpuExp);
				clSetKernelArg(context->kernel,6,sizeof(int), (void *)&where->andOr);
				clSetKernelArg(context->kernel,7,sizeof(cl_mem), (void *)&gpuFilter);
				clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);
			}

		}

		if(prevFormat == DICT){
			if(dictInit == 1){
				context->kernel = clCreateKernel(context->program,"transform_dict_filter_init",0); 
				dictInit = 0;
			}else if(dictFinal == AND)
				context->kernel = clCreateKernel(context->program,"transform_dict_filter_and",0); 
			else
				context->kernel = clCreateKernel(context->program,"transform_dict_filter_or",0); 

			clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void *) &gpuDictFilter);
			clSetKernelArg(context->kernel,1,sizeof(cl_mem), (void *) &column[prevWhere]);
			clSetKernelArg(context->kernel,2,sizeof(long), (void *) &totalTupleNum);
			clSetKernelArg(context->kernel,3,sizeof(int), (void *) &dNum);
			clSetKernelArg(context->kernel,4,sizeof(cl_mem), (void *) &gpuFilter);
			clSetKernelArg(context->kernel,5,sizeof(int), (void *) &byteNum);
			error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

			clReleaseMemObject(gpuDictFilter);
		}
	
		if(whereFree[prevWhere] == 1 && (sn->tn->dataPos[prevIndex] == MEM || sn->tn->dataPos[prevIndex] == PINNED))
			clReleaseMemObject(column[prevWhere]);

		clReleaseMemObject(gpuExp);

	}

	context->kernel = clCreateKernel(context->program, "countScanNum",0);
	clSetKernelArg(context->kernel, 0, sizeof(cl_mem), (void *) &gpuFilter);
	clSetKernelArg(context->kernel, 1, sizeof(long), (void *) &totalTupleNum);
	clSetKernelArg(context->kernel, 2, sizeof(cl_mem), (void *) &gpuCount);
	error = clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

	scanImpl(gpuCount,threadNum,gpuPsum,context, pp);

	int tmp1, tmp2;

	clEnqueueReadBuffer(context->queue, gpuCount, CL_TRUE, sizeof(int)*(threadNum-1), sizeof(int), &tmp1,0,0,0);
	clEnqueueReadBuffer(context->queue, gpuPsum, CL_TRUE, sizeof(int)*(threadNum-1), sizeof(int), &tmp2,0,0,0);

	count = tmp1+tmp2;
	res->tupleNum = count;
	printf("scanNum %d\n",count);

	clReleaseMemObject(gpuCount);

	cl_mem *result, *scanCol;

	attrNum = sn->outputNum;

	scanCol = (cl_mem*) malloc(attrNum * sizeof(cl_mem));
	result = (cl_mem*) malloc(attrNum * sizeof(cl_mem));
	
	for(int i=0;i<attrNum;i++){

		int pos = colWherePos[i];
		int index = sn->outputIndex[i];
		tupleSize += sn->tn->attrSize[index];

		if(pos != -1){
			scanCol[i] = column[pos];
		}else{
			if(sn->tn->dataPos[index] == MEM)
				scanCol[i] = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sn->tn->attrTotalSize[index], NULL, &error);
			else if (sn->tn->dataPos[index] == PINNED)
				scanCol[i] = clCreateBuffer(context->context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, sn->tn->attrTotalSize[index], NULL, &error);

			if(sn->tn->dataFormat[index] != DICT){
				if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
					clEnqueueWriteBuffer(context->queue, scanCol[i], CL_TRUE, 0, sn->tn->attrTotalSize[index],sn->tn->content[index] ,0,0,0);
				else
					scanCol[i] = (cl_mem)sn->tn->content[index];

			}else{
				if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
					clEnqueueWriteBuffer(context->queue, scanCol[i], CL_TRUE, 0, sn->tn->attrTotalSize[index]-sizeof(struct dictHeader),sn->tn->content[index]+sizeof(struct dictHeader),0,0,0);
				else
					scanCol[i] = (cl_mem) (sn->tn->content[index]+sizeof(struct dictHeader));
			}
		}

		result[i] = clCreateBuffer(context->context, CL_MEM_READ_WRITE, count * sn->tn->attrSize[index], NULL, &error); 
	}

	if(1){

		for(int i=0; i<attrNum; i++){
			int index = sn->outputIndex[i];
			int format = sn->tn->dataFormat[index];
			if(format == UNCOMPRESSED){
				if (sn->tn->attrSize[index] == sizeof(int))
					context->kernel = clCreateKernel(context->program,"scan_int",0);
				else
					context->kernel = clCreateKernel(context->program,"scan_other",0);

				clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void *)&scanCol[i]);
				clSetKernelArg(context->kernel,1,sizeof(int),(void *)&sn->tn->attrSize[index]);
				clSetKernelArg(context->kernel,2,sizeof(long),(void *)&totalTupleNum);
				clSetKernelArg(context->kernel,3,sizeof(cl_mem),(void *)&gpuPsum);
				clSetKernelArg(context->kernel,4,sizeof(long),(void *)&count);
				clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void *)&gpuFilter);
				clSetKernelArg(context->kernel,6,sizeof(cl_mem),(void *)&result[i]);
				clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

			}else if(format == DICT){
				struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
				int byteNum = dheader->bitNum/8;

				cl_mem gpuDictHeader = clCreateBuffer(context->context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, count * sn->tn->attrSize[index], dheader, &error); 

				if (sn->tn->attrSize[i] == sizeof(int))
					context->kernel = clCreateKernel(context->program,"scan_dict_int",0);
				else
					context->kernel = clCreateKernel(context->program,"scan_dict_int",0);

				clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void *)&scanCol[i]);
				clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void *)&gpuDictHeader);
				clSetKernelArg(context->kernel,2,sizeof(int),(void *)&byteNum);
				clSetKernelArg(context->kernel,3,sizeof(int),(void *)&sn->tn->attrSize[index]);
				clSetKernelArg(context->kernel,4,sizeof(long),(void *)&totalTupleNum);
				clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void *)&gpuPsum);
				clSetKernelArg(context->kernel,6,sizeof(long),(void *)&count);
				clSetKernelArg(context->kernel,7,sizeof(cl_mem),(void *)&gpuFilter);
				clSetKernelArg(context->kernel,8,sizeof(cl_mem),(void *)&result[i]);
				clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

				clReleaseMemObject(gpuDictHeader);

			}else if(format == RLE){
				int dNum = (sn->tn->attrTotalSize[index] - sizeof(struct rleHeader))/(3*sizeof(int));
				cl_mem gpuRle = clCreateBuffer(context->context, CL_MEM_READ_ONLY, totalTupleNum * sizeof(int), NULL, &error);

				long offset = 0;
				clCreateKernel(context->program,"unpack_rle",0);
				clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&scanCol[i]);
				clSetKernelArg(context->kernel,1,sizeof(cl_mem),(void*)&gpuRle);
				clSetKernelArg(context->kernel,2,sizeof(long),(void*)&totalTupleNum);
				clSetKernelArg(context->kernel,3,sizeof(long),(void*)&offset);
				clSetKernelArg(context->kernel,4,sizeof(int), (void*)&dNum);
				clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

				clCreateKernel(context->program,"scan_int",0);
				clSetKernelArg(context->kernel,0,sizeof(cl_mem),(void*)&gpuRle);
				clSetKernelArg(context->kernel,1,sizeof(int),(void*)&sn->tn->attrSize[index]);
				clSetKernelArg(context->kernel,2,sizeof(long),(void*)&totalTupleNum);
				clSetKernelArg(context->kernel,3,sizeof(cl_mem),(void*)&gpuPsum);
				clSetKernelArg(context->kernel,4,sizeof(long),(void*)&count);
				clSetKernelArg(context->kernel,5,sizeof(cl_mem),(void*)&gpuFilter);
				clSetKernelArg(context->kernel,6,sizeof(cl_mem),(void*)&result[i]);
				clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

				clReleaseMemObject(gpuRle);
			}

		}
	}

	res->tupleSize = tupleSize;

	for(int i=0;i<attrNum;i++){

		int index = sn->outputIndex[i];

		if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == PINNED)
			clReleaseMemObject(scanCol[i]);

		int colSize = res->tupleNum * res->attrSize[i];

		res->attrTotalSize[i] = colSize;
		res->dataFormat[i] = UNCOMPRESSED;

		if(sn->keepInGpu == 1){
			res->dataPos[i] = GPU;
			res->content[i] = (char *)result[i];
		}else{
			res->dataPos[i] = MEM;
			res->content[i] = (char *)malloc(colSize);
			memset(res->content[i],0,colSize);
			clEnqueueReadBuffer(context->queue, result[i], CL_TRUE, 0, colSize,res->content[i],0,0,0);
			clReleaseMemObject(result[i]);
		}
	}

	clReleaseMemObject(gpuPsum);
	clReleaseMemObject(gpuFilter);

	free(column);
	free(scanCol);
	free(result);

	clock_gettime(CLOCK_REALTIME,&end);
        double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
        printf("TableScan Time: %lf\n", timeE/(1000*1000));

	return res;

}


