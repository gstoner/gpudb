#ifndef SCAN_IMPL_CPP
#define SCAN_IMPL_CPP

#include "scan.cpp"
#include "../include/common.h"
#include "../include/gpuOpenclLib.h"
#include <CL/cl.h>

static void scanImpl(cl_mem d_input, int rLen, cl_mem d_output, struct clContext * context, struct statistic * pp)
{
	int len = 2;
	if(rLen < len){
		cl_mem input, output;
		size_t globalSize = 1;
		size_t localSize = 1;
		input = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int) * len, NULL, 0);
		output = clCreateBuffer(context->context,CL_MEM_READ_WRITE, sizeof(int) * len, NULL, 0);
		context->kernel = clCreateKernel(context->program,"cl_memset_int",0);

        	clSetKernelArg(context->kernel,0,sizeof(cl_mem), (void*)&input);
        	clSetKernelArg(context->kernel,1,sizeof(int), (void*)&len);
       		clEnqueueNDRangeKernel(context->queue, context->kernel, 1, 0, &globalSize,&localSize,0,0,0);

		clEnqueueWriteBuffer(context->queue, input, CL_TRUE, 0, rLen * sizeof(int), d_input,0,0,0);
                preallocBlockSums(len, context);
                prescanArray(output, input, len, context,pp);
                deallocBlockSums();
		clEnqueueWriteBuffer(context->queue, d_output, CL_TRUE, 0, rLen * sizeof(int), output,0,0,0);
		clReleaseMemObject(input);
		clReleaseMemObject(output);
                return;
	}else{
		preallocBlockSums(rLen, context);
		prescanArray(d_output, d_input, rLen, context,pp);
		deallocBlockSums();
	}
}


#endif

