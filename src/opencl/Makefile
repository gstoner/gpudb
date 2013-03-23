LDFLAGS   := -lrt -lm -lOpenCL

loader: load.c
	gcc -o gpuDBLoader load.c

sort: columnSort.c
	gcc -o columnSort columnSort.c -std=c99 

rle: rle.c
	gcc -std=c99 rle.c -o rleCompression

dict: dict.c
	gcc -std=c99 dict.c -o dictCompression

gpudb: gpu cpu
	 g++ -pg tableScan.o hashJoin.o materialize.o groupby.o  driver.o $(LDFLAGS) -o GPUDATABASE

cpu: driver.cpp parseKernel.cpp
	g++ -c -g driver.cpp parseKernel.cpp

gpu: tableScan.cpp hashJoin.cpp  materialize.cpp groupby.cpp 
	g++ -c -g tableScan.cpp hashJoin.cpp  materialize.cpp groupby.cpp

all: loader gpudb

clean:
	rm -rf *.o GPUDATABASE gpuDBLoader
