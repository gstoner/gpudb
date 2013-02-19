#ifndef __GPU_LIB_H_
#define __GPU_LIB_H_

#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>

#define CUDA_SAFE_CALL_NO_SYNC( call) do {                                \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#define CUDA_SAFE_CALL( call) do {                                        \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


#define CUDA_SAFE_CALL_NO_SYNC_RETRY( call) do {                                \
    cudaError err = call;                                                    \
    while( cudaSuccess != err) {                                                \
	err = call;								\
    } } while (0)

static size_t getGpuGlobalMem(int deviceID){
        size_t free = 0, total = 0;

        cudaMemGetInfo(&free,&total);
        return total;
}

static long getAvailMem(void){
	long mem = sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
	return mem;
}

static double getCurrentTime(void){
	struct timeval tv;
        gettimeofday(&tv, NULL);
        double curr  = (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000 ;
	return curr;
}

#endif
