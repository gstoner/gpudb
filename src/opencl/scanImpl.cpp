#ifndef SCAN_IMPL_CPP
#define SCAN_IMPL_CPP

#include "scan.cpp"
#include "../include/common.h"
#include "../include/gpuOpenclLib.h"

static void scanImpl(int *d_input, int rLen, int *d_output, struct clContext * context, struct statistic * pp)
{
	preallocBlockSums(rLen, context);
	prescanArray(d_output, d_input, rLen, context,pp);
	deallocBlockSums();
}


#endif

