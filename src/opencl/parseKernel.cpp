#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string>
#include <string.h>
using namespace std;

const char ** createProgram(string path, int * num){
	int count = 0;
	string * res ;

	FILE *fp = fopen(path.c_str(), "r");

	if(fp == NULL){
		printf("Failed to open kernel file\n");
		exit(-1);
	}

	char buf[128] = "";
	
	while(fgets(buf, sizeof(buf),fp)!= NULL){
		if (strstr(buf, "__kernel") != NULL){
			count ++;
		}
	}
	fclose(fp);

	*num = count;

	res = new string[count];
	const char ** ps= new const char *[count]; 

	fp = fopen(path.c_str(), "r");
	count = 0;
	while(fgets(buf, sizeof(buf),fp)!= NULL){
		if (strstr(buf, "__kernel") != NULL){
			count ++;
			res[count-1].append(buf);
		}else{
			if(count >0)
				res[count-1].append(buf);
		}
	}
	fclose(fp);

	for(int i=0;i<count;i++)
		ps[i] = res[i].c_str();

	return ps;
}

const char * createProgramBinary(string path, size_t * size){
	FILE *fp = fopen(path.c_str(),"r");

	char * res ;

	if(fp == NULL){
		printf("Failed to open binary kernel file\n");
		exit(-1);
	}

	fseek(fp,0,SEEK_END);
	*size = ftell(fp);
	rewind(fp);

	res = (char *) malloc(*size);

	if(!res){
		printf("Failed to allocate space for binary code\n");
		exit(-1);
	}

	fread(res,1,*size,fp);
	fclose(fp);

	return res;
	
}
