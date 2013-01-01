#define _FILE_OFFSET_BITS	64
#define _LARGEFILE_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <error.h>
#include "schema.h"
#include "common.h"

// write out ssb's data table in binary format
void datewrite(FILE * fp){
	struct ddate tmp;
	char data [32] = {0};
	char buf[512] = {0};
	int count = 0, i = 0,prev = 0;
	FILE * out[17];

	for(i=0;i<17;i++){
		char path[32] = {0};
		sprintf(path,"DDATE%d",i);
		out[i] = fopen(path,"w");
	}

	struct columnHeader header;
	long tupleNum = 0;
	while(fgets(buf,sizeof(buf),fp) !=NULL)
		tupleNum ++;

	header.tupleNum = tupleNum;
	header.format = UNCOMPRESSED;

	fseek(fp,0,SEEK_SET);

	for(i=0;i<17;i++){
		fwrite(&header, sizeof(struct columnHeader), 1, out[i]);
	}

	while(fgets(buf,sizeof(buf),fp)!= NULL){
		for(i = 0, prev = 0,count=0; buf[i] !='\n';i++){
			memset(data,0,sizeof(data));
			if (buf[i] == '|'){
				strncpy(data,buf+prev,i-prev);
				prev = i+1;
				switch(count){
					case 0: tmp.d_datekey = strtol(data,NULL,10); 
						fwrite(&(tmp.d_datekey),sizeof(tmp.d_datekey),1,out[0]);
						break; 
					case 1: strcpy(tmp.d_date,data); 
						fwrite(&(tmp.d_date),sizeof(tmp.d_date),1,out[1]);
						break;
					case 2: strcpy(tmp.d_dayofweek,data);
						fwrite(&(tmp.d_dayofweek),sizeof(tmp.d_dayofweek),1,out[2]);
						break;
					case 3: strcpy(tmp.d_month,data);
						fwrite(&(tmp.d_month),sizeof(tmp.d_month),1,out[3]);
						break;
					case 4: tmp.d_year = strtol(data,NULL,10);
						fwrite(&(tmp.d_year),sizeof(tmp.d_year),1,out[4]);
						break;
					case 5: tmp.d_yearmonthnum = strtol(data,NULL,10); 
						fwrite(&(tmp.d_yearmonthnum),sizeof(tmp.d_yearmonthnum),1,out[5]);
						break;
					case 6: strcpy(tmp.d_yearmonth,data);
						fwrite(&(tmp.d_yearmonth),sizeof(tmp.d_yearmonth),1,out[6]);
						break;
					case 7: tmp.d_daynuminweek = strtol(data,NULL,10); 
						fwrite(&(tmp.d_daynuminweek),sizeof(tmp.d_daynuminweek),1,out[7]);
						break;
					case 8: tmp.d_daynuminmonth = strtol(data,NULL,10); 
						fwrite(&(tmp.d_daynuminmonth),sizeof(tmp.d_daynuminmonth),1,out[8]);
						break;
					case 9: tmp.d_daynuminyear =  strtol(data,NULL,10); 
						fwrite(&(tmp.d_daynuminyear),sizeof(tmp.d_daynuminyear),1,out[9]);
						break;
					case 10: tmp.d_monthnuminyear = strtol(data,NULL,10);
						fwrite(&(tmp.d_monthnuminyear),sizeof(tmp.d_monthnuminyear),1,out[10]);
						break;
					case 11: tmp.d_weeknuminyear = strtol(data,NULL,10);
						fwrite(&(tmp.d_weeknuminyear),sizeof(tmp.d_weeknuminyear),1,out[11]);
						break;
					case 12: strcpy(tmp.d_sellingseason,data);
						fwrite(&(tmp.d_sellingseason),sizeof(tmp.d_sellingseason),1,out[12]);
						break;
					case 13: tmp.d_lastdayinweekfl[0] = data[0]; 
						fwrite(&(tmp.d_lastdayinweekfl[0]),sizeof(char),1,out[13]);
						break;
					case 14: tmp.d_lastdayinmonthfl[0] = data[0]; 
						fwrite(&(tmp.d_lastdayinmonthfl[0]),sizeof(char),1,out[14]);
						break;
					case 15: tmp.d_holidayfl[0] = data[0]; 
						fwrite(&(tmp.d_holidayfl[0]),sizeof(char),1,out[15]);
						break;
				}
				count ++;
			}
		}
		tmp.d_weekdayfl[0] = buf[i-1];
		fwrite(&(tmp.d_weekdayfl[0]),sizeof(char),1,out[16]);
	}
	for(i=0;i<17;i++){
		fclose(out[i]);
	}
}

void lineorder(FILE *fp){
	struct lineorder tmp;
	char data[32] = {0};
	char buf[512] = {0};
	int count = 0, i = 0, prev = 0;
	FILE * out[17];

	for(i=0;i<17;i++){
		char path[32] = {0};
		sprintf(path,"LINEORDER%d",i);
		out[i] = fopen(path,"w");
	}

	struct columnHeader header;
	long tupleNum = 0;
	while(fgets(buf,sizeof(buf),fp) !=NULL)
		tupleNum ++;

	header.tupleNum = tupleNum;
	header.format = UNCOMPRESSED;

	fseek(fp,0,SEEK_SET);

	for(i=0;i<17;i++)
		fwrite(&header, sizeof(struct columnHeader), 1, out[i]);


	while(fgets(buf,sizeof(buf),fp)!= NULL){
		for(i=0,prev=0,count=0;buf[i]!='\n';i++){
			memset(data,0,sizeof(data));
			if (buf[i] == '|'){
				strncpy(data,buf+prev,i-prev);
				prev = i+1;
				switch(count){
					case 0: tmp.lo_orderkey = strtol(data,NULL,10);
						fwrite(&(tmp.lo_orderkey),sizeof(tmp.lo_orderkey),1,out[0]);
						break;
					case 1: tmp.lo_linenumber = strtol(data,NULL,10);
						fwrite(&(tmp.lo_linenumber),sizeof(tmp.lo_linenumber),1,out[1]);
						break;
					case 2: tmp.lo_custkey = strtol(data,NULL,10);
						fwrite(&(tmp.lo_custkey),sizeof(tmp.lo_custkey),1,out[2]);
						break;
					case 3: tmp.lo_partkey = strtol(data,NULL,10);
						fwrite(&(tmp.lo_partkey),sizeof(tmp.lo_partkey),1,out[3]);
						break;
					case 4: tmp.lo_suppkey = strtol(data,NULL,10);
						fwrite(&(tmp.lo_suppkey),sizeof(tmp.lo_suppkey),1,out[4]);
						break;
					case 5: tmp.lo_orderdate = strtol(data,NULL,10);
						fwrite(&(tmp.lo_orderdate),sizeof(tmp.lo_orderdate),1,out[5]);
						break;
					case 6: strcpy(tmp.lo_orderpriority,data);
						fwrite(&(tmp.lo_orderpriority),sizeof(tmp.lo_orderpriority),1,out[6]);
						break;
					case 7: tmp.lo_shippriority[0] = data[0];
						fwrite(&(tmp.lo_shippriority[0]),sizeof(tmp.lo_shippriority),1,out[7]);
						break;
					case 8: tmp.lo_quantity = strtol(data,NULL,10);
						fwrite(&(tmp.lo_quantity),sizeof(tmp.lo_quantity),1,out[8]);
						break;
					case 9: tmp.lo_extendedprice = strtol(data,NULL,10);
						fwrite(&(tmp.lo_extendedprice),sizeof(tmp.lo_extendedprice),1,out[9]);
						break;
					case 10: tmp.lo_ordtotalprice = strtol(data,NULL,10);
						fwrite(&(tmp.lo_ordtotalprice),sizeof(tmp.lo_ordtotalprice),1,out[10]);
						break;
					case 11: tmp.lo_discount = strtol(data,NULL,10);
						fwrite(&(tmp.lo_discount),sizeof(tmp.lo_discount),1,out[11]);
						break;
					case 12: tmp.lo_revenue = strtol(data,NULL,10);
						fwrite(&(tmp.lo_revenue),sizeof(tmp.lo_revenue),1,out[12]);
						break;
					case 13: tmp.lo_supplycost = strtol(data,NULL,10);
						fwrite(&(tmp.lo_supplycost),sizeof(tmp.lo_supplycost),1,out[13]);
						break;
					case 14: tmp.lo_tax = strtol(data,NULL,10);
						fwrite(&(tmp.lo_tax),sizeof(tmp.lo_tax),1,out[14]);
						break;
					case 15: tmp.lo_commitdate = strtol(data,NULL,10);
						fwrite(&(tmp.lo_commitdate),sizeof(tmp.lo_commitdate),1,out[15]);
						break;
					default : break;
				}
				count ++;
			}
		}
		strncpy(tmp.lo_shipmode,buf+prev,i-prev);
		fwrite(&(tmp.lo_shipmode),sizeof(tmp.lo_shipmode),1,out[16]);
	}
	for(i=0;i<17;i++){
		fclose(out[i]);
	}
}

void part(FILE *fp){
	struct part tmp;
	char data[32] = {0};
	char buf[512] = {0};
	int count = 0, i=0, prev = 0;
	FILE * out[17];

	for(i=0;i<9;i++){
		char path[32] = {0};
		sprintf(path,"PART%d",i);
		out[i] = fopen(path,"w");
	}

	struct columnHeader header;
	long tupleNum = 0;
	while(fgets(buf,sizeof(buf),fp) !=NULL)
		tupleNum ++;

	header.tupleNum = tupleNum;
	header.format = UNCOMPRESSED;

	fseek(fp,0,SEEK_SET);

	for(i=0;i<9;i++)
		fwrite(&header, sizeof(struct columnHeader), 1, out[i]);


	while(fgets(buf,sizeof(buf),fp)!= NULL){
		for(i=0,prev=0,count=0;buf[i]!='\n';i++){
			memset(data,0,sizeof(data));
			if(buf[i] == '|'){
				strncpy(data,buf+prev,i-prev);
				prev = i+1;
				switch(count){
					case 0: tmp.p_partkey = strtol(data,NULL,10);
						fwrite(&(tmp.p_partkey),sizeof(tmp.p_partkey),1,out[0]);
						break;
					case 1: strcpy(tmp.p_name,data);
						fwrite(&(tmp.p_name),sizeof(tmp.p_name),1,out[1]);
						break;
					case 2: strcpy(tmp.p_mfgr,data);
						fwrite(&(tmp.p_mfgr),sizeof(tmp.p_mfgr),1,out[2]);
						break;
					case 3: strcpy(tmp.p_category,data);
						fwrite(&(tmp.p_category),sizeof(tmp.p_category),1,out[3]);
						break;
					case 4: strcpy(tmp.p_brand1,data);
						fwrite(&(tmp.p_brand1),sizeof(tmp.p_brand1),1,out[4]);
						break;
					case 5: strcpy(tmp.p_color,data); 
						fwrite(&(tmp.p_color),sizeof(tmp.p_color),1,out[5]);
						break;
					case 6: strcpy(tmp.p_type,data);
						fwrite(&(tmp.p_type),sizeof(tmp.p_type),1,out[6]);
						break;
					case 7: tmp.p_size = strtol(data,NULL,10);
						fwrite(&(tmp.p_size),sizeof(tmp.p_size),1,out[7]);
						break;
					default:break;
				}
				count ++;
			}
		}
		strncpy(tmp.p_container,buf+prev,i-prev);
		fwrite(&(tmp.p_container),sizeof(tmp.p_container),1,out[8]);
	}
	for(i=0;i<9;i++){
		fclose(out[i]);
	}
}

void supplier(FILE *fp){
	struct supplier tmp;
	char data[32] = {0};
	char buf[512] = {0};
	int count=0, i=0, prev = 0;
	FILE * out[7];

	for(i=0;i<7;i++){
		char path[32] = {0};
		sprintf(path,"SUPPLIER%d",i);
		out[i] = fopen(path,"w");
	}

	struct columnHeader header;
	long tupleNum = 0;
	while(fgets(buf,sizeof(buf),fp) !=NULL)
		tupleNum ++;

	header.tupleNum = tupleNum;
	header.format = UNCOMPRESSED;

	fseek(fp,0,SEEK_SET);

	for(i=0;i<7;i++)
		fwrite(&header, sizeof(struct columnHeader), 1, out[i]);


	while(fgets(buf,sizeof(buf),fp)!= NULL){
		for(i=0,prev=0,count=0;buf[i]!='\n';i++){
			memset(data,0,sizeof(data));
			if(buf[i]=='|'){
				strncpy(data,buf+prev,i-prev);
				prev = i +1;
				switch(count){
					case 0: tmp.s_suppkey = strtol(data,NULL,10);
						fwrite(&(tmp.s_suppkey),sizeof(tmp.s_suppkey),1,out[0]);
						break;
					case 1: strcpy(tmp.s_name,data);
						fwrite(&(tmp.s_name),sizeof(tmp.s_name),1,out[1]);
						break;
					case 2: strcpy(tmp.s_address,data);
						fwrite(&(tmp.s_address),sizeof(tmp.s_address),1,out[2]);
						break;
					case 3: strcpy(tmp.s_city,data);
						fwrite(&(tmp.s_city),sizeof(tmp.s_city),1,out[3]);
						break;
					case 4: strcpy(tmp.s_nation,data);
						fwrite(&(tmp.s_nation),sizeof(tmp.s_nation),1,out[4]);
						break;
					case 5: strcpy(tmp.s_region,data);
						fwrite(&(tmp.s_region),sizeof(tmp.s_region),1,out[5]);
						break;
					default: break;
				}
				count ++;
			}
		}
		strncpy(tmp.s_phone,buf+prev,i-prev);
		fwrite(&(tmp.s_phone),sizeof(tmp.s_phone),1,out[6]);
	}
	for(i=0;i<7;i++){
		fclose(out[i]);
	}
}

void customer(FILE *fp){
	struct customer tmp;
	char data[32] = {0};
	char buf[512] = {0};
	int count =0,i=0,prev = 0;
	FILE * out[8];

	for(i=0;i<8;i++){
		char path[32] = {0};
		sprintf(path,"CUSTOMER%d",i);
		out[i] = fopen(path,"w");
	}

	struct columnHeader header;
	long tupleNum = 0;
	while(fgets(buf,sizeof(buf),fp) !=NULL)
		tupleNum ++;

	header.tupleNum = tupleNum;
	header.format = UNCOMPRESSED;

	fseek(fp,0,SEEK_SET);

	for(i=0;i<8;i++)
		fwrite(&header, sizeof(struct columnHeader), 1, out[i]);


	while(fgets(buf,sizeof(buf),fp)!= NULL){
		for(i=0,prev=0,count=0;buf[i]!='\n';i++){
			memset(data,0,sizeof(data));
			if(buf[i] == '|'){
				strncpy(data,buf+prev,i-prev);
				prev = i+1;
				switch(count){
					case 0: tmp.c_custkey = strtol(data,NULL,10);
						fwrite(&(tmp.c_custkey),sizeof(tmp.c_custkey),1,out[0]);
						break;
					case 1: strcpy(tmp.c_name,data);
						fwrite(&(tmp.c_name),sizeof(tmp.c_name),1,out[1]);
						break;
					case 2: strcpy(tmp.c_address,data);
						fwrite(&(tmp.c_address),sizeof(tmp.c_address),1,out[2]);
						break;
					case 3: strcpy(tmp.c_city,data);
						fwrite(&(tmp.c_city),sizeof(tmp.c_city),1,out[3]);
						break;
					case 4: strcpy(tmp.c_nation,data);
						fwrite(&(tmp.c_nation),sizeof(tmp.c_nation),1,out[4]);
						break;
					case 5: strcpy(tmp.c_region,data);
						fwrite(&(tmp.c_region),sizeof(tmp.c_region),1,out[5]);
						break;
					case 6: strcpy(tmp.c_phone,data);
						fwrite(&(tmp.c_phone),sizeof(tmp.c_phone),1,out[6]);
						break;
					default: break;
				}
				count ++;
			}
		}
		strncpy(tmp.c_mktsegment,buf+prev,i-prev);
		fwrite(&(tmp.c_mktsegment),sizeof(tmp.c_mktsegment),1,out[7]);
	}
	for(i=0;i<8;i++){
		fclose(out[i]);
	}
}


int main(int argc, char ** argv){
	FILE * in = NULL, *out = NULL;

	if (argc !=6){
		printf("Usage error: date lo part customer supplier\n");
		exit(-1);
	}

	in = fopen(argv[1],"r");
	if (!in){
		perror(argv[1]);
		exit(-1);
	}

	datewrite(in);
	fclose(in);

	in = fopen(argv[2],"r");
	if (!in){
		perror(argv[2]);
		exit(-1);
	}
	lineorder(in);
	fclose(in);

	in = fopen(argv[3],"r");
	if (!in){
		perror(argv[3]);
		exit(-1);
	}
	part(in);
	fclose(in);

	in = fopen(argv[4],"r");
	if (!in){
		perror(argv[4]);
		exit(-1);
	}
	customer(in);
	fclose(in);

	in = fopen(argv[5],"r");
	if (!in){
		perror(argv[5]);
		exit(-1);
	}
	supplier(in);
	fclose(in);
	return 0;
}
