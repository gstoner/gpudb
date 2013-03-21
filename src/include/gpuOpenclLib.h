#ifndef __GPU_OPENCLLIB__
#define __GPU_OPENCLLIB__
	struct clContext{
		cl_context context;
		cl_command_queue queue;
		cl_program program;
		cl_kernel kernel;
	};
#endif
