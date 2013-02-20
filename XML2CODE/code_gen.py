#! /usr/bin/python
"""
   Copyright (c) 2012 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

### some assumptions:
### 1. the supported query is SSBM query
### 2. the fact table is the left child of the the join node and the dimension table is the right table
### 3. the size of the dimension table is small enough to be fit into GPU memory.
### 4. currently each thread will try to allocate its needed gpu memory before executing.
###    if it fails, it can wait until it gets the memory. There may be deadlocks here.
### 5. the current relation supported in where list is only AND, OR
### 6. the data to be grouped by and ordered by are small enough to be fit into gpu memory

import sys
import commands
import os.path
import copy
import ystree
import correlation
import config
import pickle

schema = None
keepInGpu = 1 

### we will implement different join algorithm for multi-way join in star schema benchmark
### type 0: the most straight forward way, the join op will executed one by one from a bottom up fashion
### type 1: invisiable join from abadi's paper

joinType = config.joinType 
UVA = config.UVA

def column_to_variable(col):
    res = ""
    if col.column_type in ["INTEGER","DATE"]:
        res = "int " + col.column_name.lower() + ";"
    elif col.column_type in ["DECIMAL"]:
        res = "float " + col.column_name.lower() + ";"
    elif col.column_type in ["TEXT"]:
        res = "char " + col.column_name.lower() + "[" + str(col.column_others) + "];"

    return res

def generate_schema_file():
    global schema

    schema = ystree.global_table_dict
    fo = open("schema.h","w")
    print >>fo, "#ifndef __SCHEMA_H__"
    print >>fo, "#define __SCHEMA_H__"
    for tn in schema.keys():
        print >>fo, "\tstruct " + tn.lower() + " {"
        for col in schema[tn].column_list:
            print >>fo, "\t\t" + column_to_variable(col)
        print >>fo, "\t};\n"

    print >>fo, "#endif"
    fo.close()

def generate_loader():
    global schema

    schema = ystree.global_table_dict

    fo = open("load.c","w")
    print >>fo, "#define _FILE_OFFSET_BITS       64"
    print >>fo, "#define _LARGEFILE_SOURCE"
    print >>fo, "#include <stdio.h>"
    print >>fo, "#include <stdlib.h>"
    print >>fo, "#include <error.h>"
    print >>fo, "#include <unistd.h>"
    print >>fo, "#include <string.h>"
    print >>fo, "#include <getopt.h>"
    print >>fo, "#include \"schema.h\""
    print >>fo, "#include \"common.h\""
    print >>fo, "\n"

    print >>fo, "static char delimiter = '|';"

    for tn in schema.keys():
        attrLen = len(schema[tn].column_list)

        print >>fo, "void " + tn.lower() + " (FILE *fp, char *outName){\n"

        print >>fo, "\tstruct " + tn.lower() + " tmp;"
        print >>fo, "\tchar data [64] = {0};"
        print >>fo, "\tchar buf[512] = {0};"
        print >>fo, "\tint count = 0, i = 0,prev = 0;"
        print >>fo, "\tFILE * out[" + str(attrLen) + "];\n"

        print >>fo, "\tfor(i=0;i<" + str(attrLen) + ";i++){"
        print >>fo, "\t\tchar path[64] = {0};"
        print >>fo, "\t\tsprintf(path,\"%s%d\",outName,i);"
        print >>fo, "\t\tout[i] = fopen(path, \"w\");"
        print >>fo, "\t}\n"

        print >>fo, "\tstruct columnHeader header;"
        print >>fo, "\tlong tupleNum = 0;"
        print >>fo, "\twhile(fgets(buf,sizeof(buf),fp) !=NULL)"
        print >>fo, "\t\ttupleNum ++;"

        print >>fo, "\theader.tupleNum = tupleNum;"
        print >>fo, "\theader.format = UNCOMPRESSED;"

        print >>fo, "\tfseek(fp,0,SEEK_SET);"

        print >>fo, "\tfor(i=0;i<" + str(attrLen) + ";i++){"
        print >>fo, "\t\tfwrite(&header, sizeof(struct columnHeader), 1, out[i]);"
        print >>fo, "\t}\n"

        print >>fo, "\twhile(fgets(buf,sizeof(buf),fp)!= NULL){"
        print >>fo, "\t\tfor(i = 0, prev = 0,count=0; buf[i] !='\\n';i++){"
        print >>fo, "\t\t\tmemset(data,0,sizeof(data));"
        print >>fo, "\t\t\tif (buf[i] == delimiter){"
        print >>fo, "\t\t\t\tstrncpy(data,buf+prev,i-prev);"
        print >>fo, "\t\t\t\tprev = i+1;"
        print >>fo, "\t\t\t\tswitch(count){"

        for i in range(0,attrLen-1):
            col = schema[tn].column_list[i]
            print >>fo, "\t\t\t\t\t case " + str(i) + ":"

            if col.column_type == "INTEGER" or col.column_type == "DATE":
                print >>fo, "\t\t\t\t\t\ttmp."+str(col.column_name.lower()) + " = strtol(data,NULL,10);"
                print >>fo, "\t\t\t\t\t\tfwrite(&(tmp." + str(col.column_name.lower()) + "),sizeof(int),1,out["+str(i) + "]);"
            elif col.column_type == "DECIMAL":
                print >>fo, "\t\t\t\t\t\ttmp."+str(col.column_name.lower()) + " = atof(data);"
                print >>fo, "\t\t\t\t\t\tfwrite(&(tmp." + str(col.column_name.lower()) + "),sizeof(float),1,out["+str(i) + "]);"
            elif col.column_type == "TEXT":
                print >>fo, "\t\t\t\t\t\tstrcpy(tmp." + str(col.column_name.lower()) + ",data);"
                print >>fo, "\t\t\t\t\t\tfwrite(&(tmp." + str(col.column_name.lower()) + "),sizeof(tmp." +str(col.column_name.lower()) + "), 1, out[" + str(i) + "]);"

            print >>fo, "\t\t\t\t\t\tbreak;"

        print >>fo, "\t\t\t\t}"
        print >>fo, "\t\t\t\tcount++;"

        print >>fo, "\t\t\t}"
        print >>fo, "\t\t}"

        col = schema[tn].column_list[attrLen-1]
        if col.column_type == "INTEGER" or col.column_type == "DATE":
            print >>fo, "\t\ttmp."+str(col.column_name.lower()) + " = strtol(data,NULL,10);"
            print >>fo, "\t\tfwrite(&(tmp." + str(col.column_name.lower()) + "),sizeof(int),1,out["+str(attrLen-1) + "]);"
        elif col.column_type == "DECIMAL":
            print >>fo, "\t\t\t\t\t\ttmp."+str(col.column_name.lower()) + " = atof(data);"
            print >>fo, "\t\t\t\t\t\tfwrite(&(tmp." + str(col.column_name.lower()) + "),sizeof(float),1,out["+str(i) + "]);"
        elif col.column_type == "TEXT":
            print >>fo, "\t\tstrncpy(tmp." + str(col.column_name.lower()) + ",buf+prev,i-prev);"
            print >>fo, "\t\tfwrite(&(tmp." + str(col.column_name.lower()) + "),sizeof(tmp." +str(col.column_name.lower()) + "), 1, out[" + str(attrLen-1) + "]);"
        print >>fo, "\t}\n"

        print >>fo, "\tfor(i=0;i<" + str(attrLen) + ";i++){"
        print >>fo, "\t\tfclose(out[i]);"
        print >>fo, "\t}"

        print >>fo, "\n}\n"

    print >>fo, "int main(int argc, char ** argv){\n"
    print >>fo, "\tFILE * in = NULL, *out = NULL;"
    print >>fo, "\tint table;"
    print >>fo, "\tint long_index;"

    print >>fo, "\tstruct option long_options[] = {"
    for i in range(0, len(schema.keys())):
        print >>fo, "\t\t{\"" + schema.keys()[i].lower()+ "\",required_argument,0,'" + str(i) + "'},"

    print >>fo, "\t\t{\"delimiter\",required_argument,0,'" +str(i+1) + "'}"
    print >>fo, "\t};\n"

    print >>fo, "\twhile((table=getopt_long(argc,argv,\"\",long_options,&long_index))!=-1){"
    print >>fo, "\t\tswitch(table){"
    for i in range(0, len(schema.keys())):
        print >>fo, "\t\t\tcase '" + str(i) + "':"
        print >>fo, "\t\t\t\tin = fopen(optarg,\"r\");"
        print >>fo, "\t\t\t\t" + schema.keys()[i].lower() + "(in,\"" + schema.keys()[i] + "\");"
        print >>fo, "\t\t\t\tfclose(in);"
        print >>fo, "\t\t\t\tbreak;"

    print >>fo, "\t\t\tcase '" + str(i+1) + "':"
    print >>fo, "\t\t\t\tdelimiter = optarg[0];"
    print >>fo, "\t\t\t\tbreak;"
    print >>fo, "\t\t}"
    print >>fo, "\t}\n"

    print >>fo, "\treturn 0;"

    print >>fo, "}\n"

    fo.close()

class columnAttr(object):
    type = None
    size = None

    def __init__ (self):
        self.type = ""
        self.size = 0

class JoinTranslation(object):
    dimTables = None
    factTables = None
    joinNode = None
    dimIndex = None
    factIndex = None
    outIndex = None
    outAttr = None
    outPos = None

    def __init__ (self):
        self.dimTables = []
        self.factTables = []
        self.joinNode = []
        self.dimIndex = []
        self.factIndex = []
        self.outIndex = []
        self.outAttr = []
        self.outPos = []


def __get_gb_exp__(exp,tmp_list):
    if not isinstance(exp,ystree.YFuncExp):
        return

    if exp.func_name in ["SUM","AVG","COUNT","MAX","MIN"]:
        tmp_list.append(exp)
    else:
        for x in exp.parameter_list:
            __get_gb_exp__(x,tmp_list)


def get_gbexp_list(exp_list,gb_exp_list):
    for exp in exp_list:
        if not isinstance(exp,ystree.YFuncExp):
            continue
        tmp_list = []
        __get_gb_exp__(exp,tmp_list)
        for tmp in tmp_list:
            tmp_bool = False
            for gb_exp in gb_exp_list:
                if tmp.compare(gb_exp) is True:
                    tmp_bool = True
                    break
            if tmp_bool is False:
                gb_exp_list.append(tmp)


## get the translation information for join, agg and order by 
## currently we only support star schema queries
## we assume that the dimTable is always the right child of the join node 

## fix me: what if the given query is not an SSB query

def get_tables(tree, joinAttr, aggNode, orderbyNode):

    if isinstance(tree, ystree.TableNode):
        joinAttr.factTables.append(tree)
        return

    elif isinstance(tree, ystree.OrderByNode):
        obNode = copy.deepcopy(tree)
        orderbyNode.append(obNode)
        get_tables(tree.child, joinAttr, aggNode, orderbyNode)

    elif isinstance(tree, ystree.GroupByNode):

        gbNode = copy.deepcopy(tree)
        aggNode.append(gbNode)
        get_tables(tree.child, joinAttr, aggNode, orderbyNode)

    elif isinstance(tree, ystree.TwoJoinNode):
        
        leftIndex = []
        rightIndex = []
        leftAttr = []
        rightAttr = []
        leftPos = []
        rightPos = []

        newNode = copy.deepcopy(tree)
        joinAttr.joinNode.insert(0,newNode)

        for exp in tree.select_list.tmp_exp_list:

            index = tree.select_list.tmp_exp_list.index(exp)
            if isinstance(exp,ystree.YRawColExp):
                colAttr = columnAttr()
                colAttr.type = exp.column_type
                if exp.table_name == "LEFT":

                    if joinType == 0:
                        leftIndex.append(exp.column_name)

                    elif joinType == 1:
                        newExp = ystree.__trace_to_leaf__(tree,exp,False)
                        leftIndex.append(newExp.column_name)

                    leftAttr.append(colAttr)
                    leftPos.append(index)

                elif exp.table_name == "RIGHT":

                    if joinType == 0:
                        rightIndex.append(exp.column_name)

                    elif joinType == 1:
                        newExp = ystree.__trace_to_leaf__(tree,exp,False)
                        rightIndex.append(newExp.column_name)

                    rightAttr.append(colAttr)
                    rightPos.append(index)

        outList= []
        outList.append(leftIndex)
        outList.append(rightIndex)

        outAttr = []
        outAttr.append(leftAttr)
        outAttr.append(rightAttr)

        outPos = []
        outPos.append(leftPos)
        outPos.append(rightPos)

        joinAttr.outIndex.insert(0,outList)
        joinAttr.outAttr.insert(0, outAttr)
        joinAttr.outPos.insert(0, outPos)

        pkList = tree.get_pk()
        if (len(pkList[0]) != len(pkList[1])):
            print 1/0

        if joinType == 0:
            for exp in pkList[0]:
                colIndex = 0
                if isinstance(tree.left_child, ystree.TableNode):
                    colIndex = -1
                    for tmp in tree.left_child.select_list.tmp_exp_list:
                        if exp.column_name == tmp.column_name:
                            colIndex = tree.left_child.select_list.tmp_exp_list.index(tmp)
                            break
                    if colIndex == -1:
                        print 1/0
                else:
                    colIndex = exp.column_name

        elif joinType == 1:
            for exp in pkList[0]:
                newExp = ystree.__trace_to_leaf__(tree,exp,True)
                colIndex = newExp.column_name

        joinAttr.factIndex.insert(0, colIndex)

        for exp in pkList[1]:
            colIndex = 0
            if isinstance(tree.right_child, ystree.TableNode):
                colIndex = -1
                for tmp in tree.right_child.select_list.tmp_exp_list:
                    if exp.column_name == tmp.column_name:
                        colIndex = tree.right_child.select_list.tmp_exp_list.index(tmp)
                        break
                if colIndex == -1:
                    print 1/0
            else:
                colIndex = exp.column_name
            joinAttr.dimIndex.insert(0, colIndex)

        if isinstance(tree.right_child, ystree.TableNode):
            joinAttr.dimTables.insert(0, tree.right_child)

        get_tables(tree.left_child, joinAttr, aggNode, orderbyNode)

### translate the type defined in the given schema into the supported type 
### in the translated c program. Currently the c progrma supports tree types:
### INT, FLOAT and STRING

def to_ctype(colType):

    if colType in ["INTEGER","DATE"]:
        return "INT";
    elif colType in ["TEXT"]:
        return "STRING"
    elif colType in ["DECIMAL"]:
        return "FLOAT"

### get the corresponding type length for the c program

def type_length(tn, colIndex, colType):
    if colType in ["INTEGER", "DATE"]:
        return "sizeof(int)"
    elif colType in ["TEXT"]:
        colLen = schema[tn].column_list[colIndex].column_others
        return str(colLen)
    elif colType in ["DECIMAL"]:
        return "sizeof(float)"

### fix me: doesn't support too complicated where condition

def get_where_attr(exp, whereList, relList, conList):
    if isinstance(exp, ystree.YFuncExp):
        if exp.func_name in ["AND", "OR"]:
            for x in exp.parameter_list:
                if isinstance(x, ystree.YFuncExp):
                    get_where_attr(x,whereList, relList, conList)
                elif isinstance(x, ystree.YRawColExp):
                    whereList.append(x)
        else:
            relList.append(exp.func_name)
            for x in exp.parameter_list:
                if isinstance(x, ystree.YRawColExp):
                    whereList.append(x)
                elif isinstance(x, ystree.YConsExp):
                    conList.append(x.cons_value)

    elif isinstance(exp, ystree.YRawColExp):
        whereList.append(exp)

### get each column from the original where exp list
### do not include duplicate columns

def count_whereList(wlist, tlist):

    for col in wlist:
        colExist = False
        for x in tlist:
            if x.compare(col) is True:
                colExist = True
                break
        if colExist is False:
            tlist.append(col)

    return len(tlist)

### count the nested level of the where condition

def count_whereNested(exp):
    count = 0

    if isinstance(exp, ystree.YFuncExp):
        if exp.func_name in ["AND", "OR"]:
            for x in exp.parameter_list:
                max = 0
                if isinstance(x,ystree.YFuncExp) and x.func_name in ["AND","OR"]:
                    max +=1
                    max += count_whereNest(x)
                    if max > count:
                        count = max

    return count

class mathExp:
    opName = None
    leftOp = None
    rightOp = None
    value = None

    def __init__(self):
        self.opName = None
        self.leftOp = None
        self.rightOp = None
        self.value = None

    def addOp(self, exp):

        if isinstance(exp,ystree.YRawColExp):
            self.opName = "COLUMN"
            self.value = exp.column_name
        elif isinstance(exp,ystree.YConsExp):
            self.opName = "CONS"
            self.value = exp.cons_value
        elif isinstance(exp,ystree.YFuncExp):
            self.opName = exp.func_name
            leftExp = exp.parameter_list[0]
            rightExp = exp.parameter_list[1]

            self.leftOp = mathExp()
            self.rightOp = mathExp() 
            self.leftOp.addOp(leftExp)
            self.rightOp.addOp(rightExp)

### print the mathExp in c

def printMathFunc(fo,prefix, mathFunc):

    if mathFunc.opName == "COLUMN":
        print >>fo, prefix + ".op = NOOP;" 
        print >>fo, prefix + ".opNum = 1;"
        print >>fo, prefix + ".exp = NULL;"
        print >>fo, prefix + ".opType = COLUMN;"
        print >>fo, prefix + ".opValue = " + str(mathFunc.value) + ";"
    elif mathFunc.opName == "CONS":
        print >>fo, prefix + ".op = NOOP;" 
        print >>fo, prefix + ".opNum = 1;"
        print >>fo, prefix + ".exp = NULL;"
        print >>fo, prefix + ".opType = CONS;"
        print >>fo, prefix + ".opValue = " + str(mathFunc.value) + ";"
    else:
        print >>fo, prefix + ".op = " + mathFunc.opName + ";"
        print >>fo, prefix + ".opNum = 2;"
        print >>fo, prefix + ".exp = (struct mathExp *) malloc(sizeof(struct mathExp) * 2);"
        prefix1 = prefix + ".exp[0]"
        prefix2 = prefix + ".exp[1]"
        printMathFunc(fo,prefix1,mathFunc.leftOp)
        printMathFunc(fo,prefix2,mathFunc.rightOp)

def generate_code(tree):
    fo = open("driver.cu","w")

    print >>fo, "#include <stdio.h>"
    print >>fo, "#include <stdlib.h>"
    print >>fo, "#include <sys/types.h>"
    print >>fo, "#include <sys/stat.h>"
    print >>fo, "#include <fcntl.h>"
    print >>fo, "#include <sys/mman.h>"
    print >>fo, "#include <string.h>"
    print >>fo, "#include <unistd.h>"
    print >>fo, "#include <time.h>"
    print >>fo, "#include \"common.h\""
    print >>fo, "#include \"schema.h\""
    print >>fo, "#include \"cpulib.h\""
    print >>fo, "#include \"gpulib.h\""
    print >>fo, "#define BLOCK  (1024*1024*500)\n"
    print >>fo, "extern void tableScan(struct scanNode *,struct statistic *);"
    print >>fo, "extern struct tableNode* hashJoin(struct joinNode *, struct statistic *);"
    print >>fo, "extern struct tableNode* groupBy(struct groupByNode *,struct statistic *);"
    print >>fo, "extern struct tableNode* orderBy(struct orderByNode *, struct statistic *);"
    print >>fo, "extern void materializeCol(struct materializeNode * mn, struct statistic *);"

    print >>fo, "int main(int argc, char ** argv){\n"
    
    print >>fo, "//initialize the gpu device"
    print >>fo, "\tint * tmp;"
    print >>fo, "\tcudaMalloc((void **)&tmp, 4);"
    print >>fo, "\tcudaFree(tmp);\n"

    print >>fo, "\tstruct statistic pp;"
    print >>fo, "\tpp.total = pp.kernel = 0;"

    resultNode = "result"
    joinAttr = JoinTranslation()
    aggNode = []
    orderbyNode = []
    get_tables(tree, joinAttr,aggNode, orderbyNode)

    print >>fo, "\tstruct tableNode *" + resultNode + " = (struct tableNode*) malloc(sizeof(struct tableNode));"
    print >>fo, "\tinitTable("+resultNode +");"

    for tn in joinAttr.dimTables:
        print >>fo, "\tstruct tableNode *" + tn.table_name.lower() +"Table;"

    print >>fo, "\tint outFd;"
    print >>fo, "\tint outSize;"
    print >>fo, "\tchar *outTable;"
    print >>fo, "\tlong offset;"
    print >>fo, "\tstruct columnHeader header;\n"

    for tn in joinAttr.dimTables:
        print >>fo, "\t" + tn.table_name.lower()+"Table = (struct tableNode *) malloc(sizeof(struct tableNode));" 
        totalAttr = len(tn.select_list.tmp_exp_list)
        print >>fo, "\t" + tn.table_name.lower()+"Table->totalAttr = " + str(totalAttr) + ";"
        print >>fo, "\t" + tn.table_name.lower()+"Table->attrType = (int *) malloc(sizeof(int)*"+str(totalAttr)+");"
        print >>fo, "\t" + tn.table_name.lower()+"Table->attrSize = (int *) malloc(sizeof(int)*"+str(totalAttr)+");"
        print >>fo, "\t" + tn.table_name.lower()+"Table->attrIndex = (int *) malloc(sizeof(int)*"+str(totalAttr)+");"
        print >>fo, "\t" + tn.table_name.lower()+"Table->attrTotalSize = (int *) malloc(sizeof(int)*"+str(totalAttr)+");"
        print >>fo, "\t" + tn.table_name.lower()+"Table->dataPos = (int *) malloc(sizeof(int)*"+str(totalAttr)+");"
        print >>fo, "\t" + tn.table_name.lower()+"Table->dataFormat = (int *) malloc(sizeof(int)*"+str(totalAttr)+");"
        print >>fo, "\t" + tn.table_name.lower()+"Table->content = (char **) malloc(sizeof(char *)*"+str(totalAttr)+");"

        setTupleNum = 0 
        tupleSize = "0"
        for i in range(0,totalAttr):
            col = tn.select_list.tmp_exp_list[i]
            ctype = to_ctype(col.column_type)
            colIndex = int(col.column_name)
            colLen = type_length(tn.table_name, colIndex, col.column_type) 
            tupleSize += " + " + colLen

            print >>fo, "\t" + tn.table_name.lower()+"Table->attrSize["+str(i) + "] = "+ colLen + ";"
            print >>fo, "\t" + tn.table_name.lower()+"Table->attrIndex["+str(i) + "] = "+ str(colIndex) + ";"
            print >>fo, "\t" + tn.table_name.lower()+"Table->attrType[" + str(i) + "] = " + ctype + ";"
            print >>fo, "\t" + tn.table_name.lower()+"Table->dataPos[" + str(i) + "] = MEM;"

            print >>fo, "\toutFd = open(\""+tn.table_name+str(colIndex)+"\",O_RDONLY);"
            print >>fo, "\tread(outFd,&header, sizeof(struct columnHeader));"
            print >>fo, "\toffset = sizeof(struct columnHeader);"
            print >>fo, "\t" + tn.table_name.lower() + "Table->dataFormat[" + str(i) + "] = header.format;"

            if setTupleNum == 0:
                setTupleNum = 1
                print >>fo, "\t"+tn.table_name.lower()+"Table->tupleNum = header.tupleNum;"

            print >>fo, "\toutSize = lseek(outFd,offset,SEEK_END);"
            print >>fo, "\t" + tn.table_name.lower() + "Table->attrTotalSize[" + str(i) + "] = outSize;"
            print >>fo, "\t"+tn.table_name.lower()+"Table->content["+str(i)+"] = (char *) malloc(outSize);"
            print >>fo, "\toutTable =(char *) mmap(0,outSize + offset ,PROT_READ,MAP_SHARED,outFd,0);\n"
            print >>fo, "\tmemcpy("+tn.table_name.lower()+"Table->content["+str(i)+"],outTable + offset,outSize);"
            print >>fo, "\tmunmap(outTable,outSize+offset);"
            print >>fo, "\tclose(outFd);"

        tupleSize += ";\n"
        print >>fo, "\t" + tn.table_name.lower() + "Table->tupleSize = " + tupleSize

        if tn.where_condition is not None:
            whereList = [] 
            relList = []
            conList = []

            get_where_attr(tn.where_condition.where_condition_exp, whereList, relList, conList)
            newWhereList = []
            whereLen = count_whereList(whereList, newWhereList)
            nested = count_whereNested(tn.where_condition.where_condition_exp)

            if nested != 0:
                print "Not supported yet: the where expression is too complicated"
                print 1/0

            relName = tn.table_name.lower() + "Rel"
            print >>fo, "\tstruct scanNode " + relName + ";"
            print >>fo, "\t" + relName + ".tn = " + tn.table_name.lower() + "Table;"
            print >>fo, "\t" + relName + ".hasWhere = 1;"
            print >>fo, "\t" + relName + ".whereAttrNum = " + str(whereLen) + ";"
            print >>fo, "\t" + relName + ".whereAttrType = (int *)malloc(sizeof(int)*" + str(len(whereList)) + ");"
            print >>fo, "\t" + relName + ".whereAttrSize = (int *)malloc(sizeof(int)*" + str(len(whereList)) + ");"
            print >>fo, "\t" + relName + ".whereSize = (int *)malloc(sizeof(int)*" + str(len(whereList)) + ");"
            print >>fo, "\t" + relName + ".whereIndex = (int *)malloc(sizeof(int)*" + str(len(whereList)) + ");"
            print >>fo, "\t" + relName + ".whereFormat = (int *)malloc(sizeof(int)*" + str(len(whereList)) + ");"
            print >>fo, "\t" + relName + ".wherePos = (int *)malloc(sizeof(int)*" + str(len(whereList)) + ");"
            print >>fo, "\t" + relName + ".content = (char **)malloc(sizeof(char *)*" + str(len(whereList)) + ");"
            if keepInGpu == 0:
                print >>fo, "\t" + relName + ".keepInGpu = 0;"
            else:
                print >>fo, "\t" + relName + ".keepInGpu = 1;"

            for i in range(0,len(newWhereList)):
                colIndex = int(newWhereList[i].column_name)
                colType = schema[newWhereList[i].table_name].column_list[colIndex].column_type
                colLen = type_length(newWhereList[i].table_name,colIndex,colType)
                ctype = to_ctype(colType)
                print >>fo, "\t" + relName + ".whereAttrType["+str(i) + "] = " + ctype + ";"
                print >>fo, "\t" + relName + ".whereAttrSize["+str(i) + "] = " + str(colLen) + ";"
                print >>fo, "\t" + relName + ".whereIndex["+str(i) + "] = " + str(colIndex) + ";"
                print >>fo, "\toutFd = open(\""+tn.table_name+str(colIndex)+"\",O_RDONLY);"
                print >>fo, "\tread(outFd,&header, sizeof(struct columnHeader));"
                print >>fo, "\t" + relName + ".whereFormat[" + str(i) + "] = header.format;"
                print >>fo, "\t" + relName + ".wherePos[" + str(i) + "] = MEM;"
                print >>fo, "\toffset = sizeof(struct columnHeader);"
                print >>fo, "\toutSize = lseek(outFd,offset,SEEK_END);"
                print >>fo, "\t" + relName + ".whereSize[" + str(i) + "] = outSize;"
                print >>fo, "\t"+relName+".content["+str(i)+"] = (char *) malloc(outSize);"
                print >>fo, "\toutTable =(char *) mmap(0,outSize + offset,PROT_READ,MAP_SHARED,outFd,0);\n"
                print >>fo, "\tmemcpy("+relName+".content["+str(i)+"],outTable + offset,outSize);"
                print >>fo, "\tmunmap(outTable,outSize + offset);"
                print >>fo, "\tclose(outFd);"

            print >>fo, "\t" + relName + ".filter = (struct whereCondition *)malloc(sizeof(struct whereCondition));"

            print >>fo, "\t(" + relName + ".filter)->nested = 0;"
            print >>fo, "\t(" + relName + ".filter)->expNum = " + str(len(whereList)) + ";"
            print >>fo, "\t(" + relName + ".filter)->exp = (struct whereExp*) malloc(sizeof(struct whereExp) *" + str(len(whereList)) + ");"

            if tn.where_condition.where_condition_exp.func_name in ["AND","OR"]:
                print >>fo, "\t(" + relName + ".filter)->andOr = " + tn.where_condition.where_condition_exp.func_name + ";"

            else:
                print >>fo, "\t(" + relName + ".filter)->andOr = EXP;"

            for i in range(0,len(whereList)):
                colIndex = -1
                for j in range(0,len(newWhereList)):
                    if newWhereList[j].compare(whereList[i]) is True:
                        colIndex = j
                        break

                if colIndex <0:
                    print 1/0

                print >>fo, "\t(" + relName + ".filter)->exp[" + str(i) + "].index = " + str(colIndex) + ";"
                print >>fo, "\t(" + relName + ".filter)->exp[" + str(i) + "].relation = " + relList[i] + ";" 

                colType = whereList[i].column_type
                ctype = to_ctype(colType)

                if ctype == "INT":
                    print >>fo, "\t{"
                    print >>fo, "\t\tint tmp = " + conList[i] + ";"
                    print >>fo, "\t\tmemcpy((" + relName + ".filter)->exp[" + str(i) + "].content, &tmp,sizeof(int));"
                    print >>fo, "\t}"

                elif ctype == "FLOAT":
                    print >>fo, "\t{"
                    print >>fo, "\t\tfloat tmp = " + conList[i] + ";"
                    print >>fo, "\t\tmemcpy((" + relName + ".filter)->exp[" + str(i) + "].content, &tmp,sizeof(float));"
                    print >>fo, "\t}"

                else:
                    print >>fo, "\tstrcpy((" + relName + ".filter)->exp[" + str(i) + "].content," + conList[i] + ");\n"

            print >>fo, "\ttableScan(&" + relName + ", &pp);"
            print >>fo, "\tfreeScan(&" + relName + ");\n"

############### facttabe starts ###############

    if joinType == 0:

    #### the type 0 join  starts ###

        selectOnly = len(joinAttr.dimTables) == 0
        factName = joinAttr.factTables[0].table_name.lower() + "Table"
        totalAttr = len(joinAttr.factTables[0].select_list.tmp_exp_list)
        print >>fo, "\tstruct tableNode *" + factName + " = (struct tableNode*)malloc(sizeof(struct tableNode));" 
        print >>fo, "\t" + factName + "->totalAttr = " + str(totalAttr) + ";"
        print >>fo, "\t" + factName + "->attrType = (int *) malloc(sizeof(int)*" + str(totalAttr) + ");"
        print >>fo, "\t" + factName + "->attrSize = (int *) malloc(sizeof(int)*" + str(totalAttr) + ");"
        print >>fo, "\t" + factName + "->attrIndex = (int *) malloc(sizeof(int)*" + str(totalAttr) + ");"
        print >>fo, "\t" + factName + "->attrTotalSize = (int *) malloc(sizeof(int)*" + str(totalAttr) + ");"
        print >>fo, "\t" + factName + "->dataPos = (int *) malloc(sizeof(int)*" + str(totalAttr) + ");"
        print >>fo, "\t" + factName + "->dataFormat = (int *) malloc(sizeof(int)*" + str(totalAttr) + ");"
        print >>fo, "\t" + factName + "->content = (char **) malloc(sizeof(char *)*" + str(totalAttr) + ");"

        tupleSize = "0"
        setTupleNum = 0
        for i in range(0,totalAttr):
            col = joinAttr.factTables[0].select_list.tmp_exp_list[i]
            if isinstance(col, ystree.YRawColExp):
                colType = col.column_type
                colIndex = col.column_name
                ctype = to_ctype(colType)
                colLen = type_length(joinAttr.factTables[0].table_name, colIndex, colType)
            elif isinstance(col, ystree.YConsExp):
                colType = col.cons_type
                ctype = to_ctype(colType)
                if cons_type == "INTEGER":
                    colLen = "sizeof(int)"
                elif cons_type == "FLOAT":
                    colLen = "sizeof(float)"
                else:
                    colLen = str(len(col.cons_value))
            elif isinstance(col, ystree.YFuncExp):
                print 1/0

            if setTupleNum == 0:
                setTupleNum = 1
                print >>fo, "\toutFd = open(\"" + joinAttr.factTables[0].table_name + str(colIndex) + "\",O_RDONLY);"
                print >>fo, "\tread(outFd, &header, sizeof(struct columnHeader));"
                print >>fo, "\t" + factName + "->tupleNum = header.tupleNum;"
                print >>fo, "\tclose(outFd);"

            tupleSize += " + " + colLen
            print >>fo, "\t" + factName + "->attrType[" + str(i) + "] = " + ctype + ";"
            print >>fo, "\t" + factName + "->attrSize[" + str(i) + "] = " + colLen + ";"
            print >>fo, "\t" + factName + "->attrIndex[" + str(i) + "] = " + str(colIndex) + ";"

            if UVA == 0:
                print >>fo, "\t" + factName + "->dataPos[" + str(i) + "] = MEM;"
            else:
                print >>fo, "\t" + factName + "->dataPos[" + str(i) + "] = UVA;"

        tupleSize += ";\n"
        print >>fo, "\t" + factName + "->tupleSize = " + tupleSize 

        print >>fo, "\tint pass = " + factName + "->tupleNum / BLOCK + 1;"
        print >>fo, "\tlong tupleUnit = " + factName + "->tupleNum / pass;"
        print >>fo, "\tlong nextScan = tupleUnit;"
        print >>fo, "\tlong restTuple = " + factName + "->tupleNum;"
        print >>fo, "\tlong tupleOffset = 0;"
        print >>fo, "\toffset = 0;\n"

        print >>fo, "\tfor(int i=0;i<pass;i++){\n"
        print >>fo, "\t\tif(restTuple < nextScan)"
        print >>fo, "\t\t\tnextScan = restTuple;\n"
        print >>fo, "\t\t" + factName + "->tupleNum = nextScan;"

        for i in range(0,totalAttr):
            col = joinAttr.factTables[0].select_list.tmp_exp_list[i]
            colType = col.column_type
            colIndex =  col.column_name
            colLen = type_length(joinAttr.factTables[0].table_name, colIndex, colType)

            print >>fo, "\t\toutFd = open(\"" + joinAttr.factTables[0].table_name + str(colIndex) + "\", O_RDONLY);"
            print >>fo, "\t\tread(outFd, &header, sizeof(struct columnHeader));"
            print >>fo, "\t\t" + factName + "->dataFormat[" + str(i) + "] = header.format;"
            print >>fo, "\t\tif(header.format == UNCOMPRESSED){"
            print >>fo, "\t\t\toffset = tupleOffset *" + colLen + "+sizeof(struct columnHeader);"
            print >>fo, "\t\t\toutSize = nextScan*" + colLen + ";"
            print >>fo, "\t\t\toutTable = (char *)mmap(0,outSize+offset,PROT_READ,MAP_SHARED,outFd,0);"

            if UVA == 0:
                print >>fo, "\t\t\t" + factName + "->content[" + str(i) + "] = (char *) malloc(outSize);\n"
            else:
                print >>fo, "\t\t\tCUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void**)&"+factName+"->content["+str(i)+"],outSize));"

            print >>fo, "\t\t\tmemcpy(" + factName + "->content[" + str(i) + "], outTable+offset, outSize);"
            print >>fo, "\t\t\tmunmap(outTable,outSize + offset);"
            print >>fo, "\t\t\tclose(outFd);"
            print >>fo, "\t\t\t" + factName + "->attrTotalSize[" + str(i) + "] = outSize;"
            print >>fo, "\t\t}else if (header.format == DICT){"
            print >>fo, "\t\t\tstruct dictHeader dheader;"
            print >>fo, "\t\t\tread(outFd, &dheader, sizeof(struct dictHeader));"
            print >>fo, "\t\t\toffset = tupleOffset * dheader.bitNum / 8 + sizeof(struct columnHeader) + sizeof(struct dictHeader);"
            print >>fo, "\t\t\toutSize = nextScan * dheader.bitNum / 8 + sizeof(struct dictHeader);"
            print >>fo, "\t\t\toutTable = (char *)mmap(0,outSize+offset,PROT_READ,MAP_SHARED,outFd,0);"

            if UVA == 0:
                print >>fo, "\t\t\t" + factName + "->content[" + str(i) + "] = (char *) malloc(outSize);\n"
            else:
                print >>fo, "\t\t\tCUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void**)&"+factName+"->content["+str(i)+"],outSize));"

            print >>fo, "\t\t\tmemcpy(" + factName + "->content[" + str(i) + "], &dheader, sizeof(struct dictHeader));"
            print >>fo, "\t\t\tmemcpy(" + factName + "->content[" + str(i) + "] + sizeof(struct dictHeader), outTable+offset, outSize - sizeof(struct dictHeader));"
            print >>fo, "\t\t\tmunmap(outTable,outSize + offset);"
            print >>fo, "\t\t\tclose(outFd);"
            print >>fo, "\t\t\t" + factName + "->attrTotalSize[" + str(i) + "] = outSize;"
            print >>fo, "\t\t}else if (header.format == RLE){"
            print >>fo, "\t\t\t" + factName + "->offset = tupleOffset;"
            print >>fo, "\t\t\toffset = sizeof(struct columnHeader);"
            print >>fo, "\t\t\toutSize = lseek(outFd, 0, SEEK_END) - offset;"
            print >>fo, "\t\t\toutTable = (char *)mmap(0,outSize+offset,PROT_READ,MAP_SHARED,outFd,0);"

            if UVA == 0:
                print >>fo, "\t\t\t" + factName + "->content[" + str(i) + "] = (char *) malloc(outSize);\n"
            else:
                print >>fo, "\t\t\tCUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void**)&"+factName+"->content["+str(i)+"],outSize));"

            print >>fo, "\t\t\tmemcpy(" + factName + "->content[" + str(i) + "], outTable+offset, outSize);"
            print >>fo, "\t\t\tmunmap(outTable,outSize + offset);"
            print >>fo, "\t\t\tclose(outFd);"
            print >>fo, "\t\t\t" + factName + "->attrTotalSize[" + str(i) + "] = outSize;"
            print >>fo, "\t\t}"

        if joinAttr.factTables[0].where_condition is not None:
            whereExp = joinAttr.factTables[0].where_condition.where_condition_exp
            whereList = []
            relList = []
            conList = []

            get_where_attr(whereExp,whereList,relList,conList)
            newWhereList = []
            whereLen = count_whereList(whereList, newWhereList)
            nested = count_whereNested(whereExp)

            if nested !=0:
                print "Not supported yet: the where expression is too complicated"
                print 1/0

            relName = joinAttr.factTables[0].table_name.lower() + "Rel"
            print >>fo, "\t\tstruct scanNode " + relName + ";"
            print >>fo, "\t\t" + relName + ".tn = " + factName + ";"
            print >>fo, "\t\t" + relName + ".hasWhere = 1;"
            print >>fo, "\t\t" + relName + ".whereAttrNum = " + str(whereLen) + ";"
            print >>fo, "\t\t" + relName + ".whereAttrType = (int *)malloc(sizeof(int)*" + str(whereLen) + ");"
            print >>fo, "\t\t" + relName + ".whereAttrSize = (int *)malloc(sizeof(int)*" + str(whereLen) + ");"
            print >>fo, "\t\t" + relName + ".whereSize = (int *)malloc(sizeof(int)*" + str(whereLen) + ");"
            print >>fo, "\t\t" + relName + ".whereIndex = (int *)malloc(sizeof(int)*" + str(whereLen) + ");"
            print >>fo, "\t\t" + relName + ".whereFormat = (int *)malloc(sizeof(int)*" + str(whereLen) + ");"
            print >>fo, "\t\t" + relName + ".wherePos = (int *)malloc(sizeof(int)*" + str(whereLen) + ");"
            print >>fo, "\t\t" + relName + ".content = (char **)malloc(sizeof(char *)*" + str(whereLen) + ");"
            if keepInGpu == 0:
                print >>fo, "\t\t" + relName + ".keepInGpu = 0;"
            else:
                print >>fo, "\t\t" + relName + ".keepInGpu = 1;"

            for i in range(0,len(newWhereList)):
                colIndex = int(newWhereList[i].column_name)
                colType = schema[newWhereList[i].table_name].column_list[colIndex].column_type
                colLen = type_length(newWhereList[i].table_name,colIndex,colType)
                ctype = to_ctype(colType)
                print >>fo, "\t\t" + relName + ".whereAttrType["+str(i) + "] = " + ctype + ";"
                print >>fo, "\t\t" + relName + ".whereAttrSize["+str(i) + "] = " + str(colLen) + ";"
                print >>fo, "\t\t" + relName + ".whereIndex["+str(i) + "] = " + str(colIndex) + ";"
                print >>fo, "\t\toutFd = open(\""+joinAttr.factTables[0].table_name+str(colIndex)+"\",O_RDONLY);"
                print >>fo, "\t\tread(outFd, &header, sizeof(struct columnHeader));"
                print >>fo, "\t\t" + relName + ".whereFormat[" + str(i) + "] = header.format;"
                print >>fo, "\t\tif(header.format == UNCOMPRESSED){"
                print >>fo, "\t\t\toffset = tupleOffset *" + colLen + " + sizeof(struct columnHeader);"
                print >>fo, "\t\t\toutSize = nextScan *" + colLen + ";"

                if UVA == 0:
                    print >>fo, "\t\t\t"+relName+".content["+str(i)+"] = (char *) malloc(outSize);"
                    print >>fo, "\t\t\t" + relName + ".wherePos[" + str(i) + "] = MEM;"
                else:
                    print >>fo, "\t\t\t" + relName + ".wherePos[" + str(i) + "] = UVA;"
                    print >>fo, "\t\t\tCUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void**)&"+relName+".content["+str(i)+"],outSize));"

                print >>fo, "\t\t\toutTable =(char *) mmap(0,outSize + offset,PROT_READ,MAP_SHARED,outFd,0);\n"
                print >>fo, "\t\t\tmemcpy("+relName+".content["+str(i)+"],outTable + offset,outSize);"
                print >>fo, "\t\t\tmunmap(outTable, outSize + offset);"
                print >>fo, "\t\t\tclose(outFd);"
                print >>fo, "\t\t\t"+relName+".whereSize["+str(i)+"] = outSize;"
                print >>fo, "\t\t}else if(header.format == DICT){"
                print >>fo, "\t\t\tstruct dictHeader dheader;"
                print >>fo, "\t\t\tread(outFd, &dheader, sizeof(struct dictHeader));"
                print >>fo, "\t\t\toffset = tupleOffset * dheader.bitNum / 8 + sizeof(struct columnHeader) + sizeof(struct dictHeader);"
                print >>fo, "\t\t\toutSize = nextScan * dheader.bitNum / 8 + sizeof(struct dictHeader);"
                print >>fo, "\t\t\toutTable = (char *)mmap(0,outSize+offset,PROT_READ,MAP_SHARED,outFd,0);"
                print >>fo, "\t\t\t"+relName+".content["+str(i)+"] = (char *) malloc(outSize);\n"
                print >>fo, "\t\t\tmemcpy("+relName+".content["+str(i)+"],&dheader, sizeof(struct dictHeader));"
                print >>fo, "\t\t\tmemcpy("+relName+".content["+str(i)+"] + sizeof(dictHeader),outTable + offset,outSize - sizeof(struct dictHeader));"
                print >>fo, "\t\t\tmunmap(outTable, outSize + offset);"
                print >>fo, "\t\t\tclose(outFd);"
                print >>fo, "\t\t\t"+relName+".whereSize["+str(i)+"] = outSize;"
                print >>fo, "\t\t}else if (header.format == RLE){"
                print >>fo, "\t\t\t" + relName + ".offset = tupleOffset;"
                print >>fo, "\t\t\toffset = sizeof(struct columnHeader);"
                print >>fo, "\t\t\toutSize = lseek(outFd,0,SEEK_END) - offset;"
                print >>fo, "\t\t\toutTable = (char *)mmap(0,outSize+offset,PROT_READ,MAP_SHARED,outFd,0);"
                print >>fo, "\t\t\t"+relName+".content["+str(i)+"] = (char *) malloc(outSize);"
                print >>fo, "\t\t\toutTable =(char *) mmap(0,outSize + offset,PROT_READ,MAP_SHARED,outFd,0);\n"
                print >>fo, "\t\t\tmemcpy("+relName+".content["+str(i)+"],outTable + offset,outSize);"
                print >>fo, "\t\t\tmunmap(outTable, outSize + offset);"
                print >>fo, "\t\t\tclose(outFd);"
                print >>fo, "\t\t\t"+relName+".whereSize["+str(i)+"] = outSize;"
                print >>fo, "\t\t}"

            print >>fo, "\t\t" + relName + ".filter = (struct whereCondition *)malloc(sizeof(struct whereCondition));"

            print >>fo, "\t\t(" + relName + ".filter)->nested = 0;"
            print >>fo, "\t\t(" + relName + ".filter)->expNum = " + str(len(whereList)) + ";"
            print >>fo, "\t\t(" + relName + ".filter)->exp = (struct whereExp*) malloc(sizeof(struct whereExp) *" + str(len(whereList)) + ");"

            if joinAttr.factTables[0].where_condition.where_condition_exp.func_name in ["AND","OR"]:
                print >>fo, "\t\t(" + relName + ".filter)->andOr = " + joinAttr.factTables[0].where_condition.where_condition_exp.func_name + ";"

            else:
                print >>fo, "\t\t(" + relName + ".filter)->andOr = EXP;"

            for i in range(0,len(whereList)):
                colIndex = -1
                for j in range(0,len(newWhereList)):
                    if newWhereList[j].compare(whereList[i]) is True:
                        colIndex = j
                        break

                if colIndex <0:
                    print 1/0

                print >>fo, "\t\t(" + relName + ".filter)->exp[" + str(i) + "].index = " + str(colIndex) + ";"
                print >>fo, "\t\t(" + relName + ".filter)->exp[" + str(i) + "].relation = " + relList[i] + ";" 

                colType = whereList[i].column_type
                ctype = to_ctype(colType)

                if ctype == "INT":
                    print >>fo, "\t\t{"
                    print >>fo, "\t\t\tint tmp = " + conList[i] + ";"
                    print >>fo, "\t\t\tmemcpy((" + relName + ".filter)->exp[" + str(i) + "].content, &tmp,sizeof(int));"
                    print >>fo, "\t\t}"

                elif ctype == "FLOAT":

                    print >>fo, "\t\t{"
                    print >>fo, "\t\t\tfloat tmp = " + conList[i] + ";"
                    print >>fo, "\t\t\tmemcpy((" + relName + ".filter)->exp[" + str(i) + "].content, &tmp,sizeof(float));"
                    print >>fo, "\t\t}"
                    print 1/0
                else:
                    print >>fo, "\t\tstrcpy((" + relName + ".filter)->exp[" + str(i) + "].content," + conList[i] + ");\n"

            print >>fo, "\t\ttableScan(&" + relName + ", &pp);"
            if selectOnly == 0:
                print >>fo, "\t\tfreeScan(&" + relName + ");\n"


        for i in range(0,len(joinAttr.dimTables)):
            jName = "jNode" + str(i)
            dimName = joinAttr.dimTables[i].table_name.lower() + "Table" 
            print >>fo, "\t\tstruct joinNode " + jName + ";"
            print >>fo, "\t\t" + jName + ".leftTable = " + factName + ";"
            print >>fo, "\t\t" + jName + ".rightTable = " + dimName + ";"

            lOutList = joinAttr.outIndex[i][0]
            rOutList = joinAttr.outIndex[i][1]

            lPosList = joinAttr.outPos[i][0]
            rPosList = joinAttr.outPos[i][1]

            lAttrList = joinAttr.outAttr[i][0]
            rAttrList = joinAttr.outAttr[i][1]

            print >>fo, "\t\t" + jName + ".totalAttr = " + str(len(rOutList) + len(lOutList)) + ";"
            print >>fo, "\t\t" + jName + ".keepInGpu = (int *) malloc(sizeof(int) * " + str(len(rOutList) + len(lOutList)) + ");"

            if keepInGpu == 0:
                print >>fo, "\t\tfor(int k=0;k<" + str(len(rOutList) + len(lOutList))  + ";k++)"
                print >>fo, "\t\t\t" + jName + ".keepInGpu[k] = 0;"
            else:
                print >>fo, "\t\tfor(int k=0;k<" + str(len(rOutList) + len(lOutList))  + ";k++)"
                print >>fo, "\t\t\t" + jName + ".keepInGpu[k] = 1;"

            print >>fo, "\t\t" + jName + ".rightOutputAttrNum = " + str(len(rOutList)) + ";"
            print >>fo, "\t\t" + jName + ".leftOutputAttrNum = " + str(len(lOutList)) + ";"
            print >>fo, "\t\t" + jName + ".leftOutputAttrType = (int *)malloc(sizeof(int)*" + str(len(lOutList)) + ");"
            print >>fo, "\t\t" + jName + ".leftOutputIndex = (int *)malloc(sizeof(int)*" + str(len(lOutList)) + ");"
            print >>fo, "\t\t" + jName + ".leftPos = (int *)malloc(sizeof(int)*" + str(len(lOutList)) + ");"
            print >>fo, "\t\t" + jName + ".tupleSize = 0;"
            for j in range(0,len(lOutList)):
                ctype = to_ctype(lAttrList[j].type)
                print >>fo, "\t\t" + jName + ".leftOutputIndex[" + str(j) + "] = " + str(lOutList[j]) + ";"
                print >>fo, "\t\t" + jName + ".leftOutputAttrType[" + str(j) + "] = " + ctype + ";" 
                print >>fo, "\t\t" + jName + ".leftPos[" + str(j) + "] = " + str(lPosList[j]) + ";"
                print >>fo, "\t\t" + jName + ".tupleSize += " + factName + "->attrSize[" + str(lOutList[j]) + "];"

            print >>fo, "\t\t" + jName + ".rightOutputAttrType = (int *)malloc(sizeof(int)*" + str(len(rOutList)) + ");"
            print >>fo, "\t\t" + jName + ".rightOutputIndex = (int *)malloc(sizeof(int)*" + str(len(rOutList)) + ");"
            print >>fo, "\t\t" + jName + ".rightPos = (int *)malloc(sizeof(int)*" + str(len(rOutList)) + ");"
            for j in range(0,len(rOutList)):
                ctype = to_ctype(rAttrList[j].type)
                print >>fo, "\t\t" + jName + ".rightOutputIndex[" + str(j) + "] = " + str(rOutList[j]) + ";"
                print >>fo, "\t\t" + jName + ".rightOutputAttrType[" + str(j) + "] = " + ctype + ";" 
                print >>fo, "\t\t" + jName + ".rightPos[" + str(j) + "] = " + str(rPosList[j]) + ";"
                print >>fo, "\t\t" + jName + ".tupleSize += " + dimName + "->attrSize[" + str(rOutList[j]) + "];"

            print >>fo, "\t\t" + jName + ".rightKeyIndex = " + str(joinAttr.dimIndex[i]) + ";"
            print >>fo, "\t\t" + jName + ".leftKeyIndex = " + str(joinAttr.factIndex[i]) + ";"

            print >>fo, "\t\tstruct tableNode *join" + str(i) + " = hashJoin(&" + jName + ",&pp);\n" 
            factName = "join" + str(i)

        if selectOnly == 0:
            print >>fo, "\t\tif(pass !=1){"
            print >>fo, "\t\t\tmergeIntoTable("+resultNode+",join" + str(i) + ", &pp);"
            for i in range(0,len(joinAttr.dimTables)):
                jName = "join" + str(i)
                print >>fo, "\t\t\tfreeTable(" + jName + ");"

            print >>fo, "\t\t}else{"
            print >>fo, "\t\t\t"+resultNode+" = join" + str(i) + ";" 
            for i in range(0,len(joinAttr.dimTables)-1):
                jName = "join" + str(i)
                print >>fo, "\t\t\tfreeTable(" + jName + ");"
            print >>fo, "\t\t}"

            tmpName = joinAttr.factTables[0].table_name.lower() + "Table"
            for i in range(0, totalAttr):
                print >>fo, "\t\tif(" + tmpName + "->dataPos[" + str(i) + "] == MEM)"
                print >>fo, "\t\t\tfree(" + tmpName + "->content[" + str(i) + "]);"
                print >>fo, "\t\tif(" + tmpName + "->dataPos[" + str(i) + "] == UVA)"
                print >>fo, "\t\t\tcudaFreeHost(" + tmpName + "->content[" + str(i) + "]);"
                print >>fo, "\t\telse"
                print >>fo, "\t\t\tcudaFree(" + tmpName + "->content[" + str(i) + "]);"
        else:
            print >>fo, "\t\tif(pass !=1){"
            print >>fo, "\t\t\tmergeIntoTable("+resultNode+"," + relName + ".tn, &pp);"

            tmpName = joinAttr.factTables[0].table_name.lower() + "Table"
            for i in range(0, totalAttr):
                print >>fo, "\t\t\tif(" + tmpName + "->dataPos[" + str(i) + "] == MEM)"
                print >>fo, "\t\t\t\tfree(" + tmpName + "->content[" + str(i) + "]);"
                print >>fo, "\t\t\telse if("+ tmpName + "->dataPos[" + str(i) + "] == UVA)"
                print >>fo, "\t\t\t\tcudaFreeHost(" + tmpName + "->content[" + str(i) + "]);"
                print >>fo, "\t\t\telse"
                print >>fo, "\t\t\t\tcudaFree(" + tmpName + "->content[" + str(i) + "]);"

            print >>fo, "\t\t}else{"
            print >>fo, "\t\t\t"+resultNode+" = " + relName + ".tn;" 
            print >>fo, "\t\t}"
            print >>fo, "\t\tfreeScan(&" + relName + ");\n"


        print >>fo, "\t\ttupleOffset += tupleUnit;"
        print >>fo, "\t\trestTuple -=nextScan;"
        print >>fo, "\t}\n"

    ### the type 0 join ends ###

    elif joinType == 1:

    ### the type 1 join starts ###

        factName = joinAttr.factTables[0].table_name.lower() + "Table"
        totalAttr = len(joinAttr.factTables[0].select_list.tmp_exp_list)
        print >>fo, "\tstruct tableNode *" + factName + " = (struct tableNode*)malloc(sizeof(struct tableNode));" 
        print >>fo, "\t" + factName + "->totalAttr = " + str(totalAttr) + ";"
        print >>fo, "\t" + factName + "->attrType = (int *) malloc(sizeof(int)*" + str(totalAttr) + ");"
        print >>fo, "\t" + factName + "->attrSize = (int *) malloc(sizeof(int)*" + str(totalAttr) + ");"
        print >>fo, "\t" + factName + "->attrIndex = (int *) malloc(sizeof(int)*" + str(totalAttr) + ");"
        print >>fo, "\t" + factName + "->attrTotalSize = (int *) malloc(sizeof(int)*" + str(totalAttr) + ");"
        print >>fo, "\t" + factName + "->dataPos = (int *) malloc(sizeof(int)*" + str(totalAttr) + ");"
        print >>fo, "\t" + factName + "->dataFormat = (int *) malloc(sizeof(int)*" + str(totalAttr) + ");"
        print >>fo, "\t" + factName + "->content = (char **) malloc(sizeof(char *)*" + str(totalAttr) + ");"

        tupleSize = "0"
        setTupleNum = 0
        for i in range(0,totalAttr):
            col = joinAttr.factTables[0].select_list.tmp_exp_list[i]
            if isinstance(col, ystree.YRawColExp):
                colType = col.column_type
                colIndex = col.column_name
                ctype = to_ctype(colType)
                colLen = type_length(joinAttr.factTables[0].table_name, colIndex, colType)
            elif isinstance(col, ystree.YConsExp):
                colType = col.cons_type
                ctype = to_ctype(colType)
                if cons_type == "INTEGER":
                    colLen = "sizeof(int)"
                elif cons_type == "FLOAT":
                    colLen = "sizeof(float)"
                else:
                    colLen = str(len(col.cons_value))
            elif isinstance(col, ystree.YFuncExp):
                print 1/0

            if setTupleNum == 0:
                setTupleNum = 1
                print >>fo, "\toutFd = open(\"" + joinAttr.factTables[0].table_name + str(colIndex) + "\",O_RDONLY);"
                print >>fo, "\tread(outFd,&header,sizeof(columnHeader));"
                print >>fo, "\t" + factName + "->tupleNum = header.tupleNum;" 
                print >>fo, "\tclose(outFd);"

            tupleSize += " + " + colLen
            print >>fo, "\t" + factName + "->attrType[" + str(i) + "] = " + ctype + ";"
            print >>fo, "\t" + factName + "->attrSize[" + str(i) + "] = " + colLen + ";"
            print >>fo, "\t" + factName + "->attrIndex[" + str(i) + "] = " + str(colIndex) + ";"
            print >>fo, "\t" + factName + "->dataPos[" + str(i) + "] = MEM;"

        tupleSize += ";\n"
        print >>fo, "\t" + factName + "->tupleSize = " + tupleSize

        factIndex = []
        factInputList = joinAttr.factTables[0].select_list.tmp_exp_list
        dimNum = len(joinAttr.dimTables)
        outputList = joinAttr.joinNode[dimNum-1].select_list.tmp_exp_list
        outputNum =  len(outputList)

        jName = "jNode"
        print >>fo, "\tstruct joinNode " + jName + ";"
        print >>fo, "\t" + jName + ".dimNum = " + str(dimNum) + ";" 
        print >>fo, "\t" + jName + ".factTable = " + factName + ";"
        print >>fo, "\t" + jName + ".dimTable = (struct tableNode **) malloc(sizeof(struct tableNode) * " + jName + ".dimNum);"
        print >>fo, "\t" + jName + ".factIndex = (int *) malloc(sizeof(int) * " + jName + ".dimNum);"
        print >>fo, "\t" + jName + ".dimIndex = (int *) malloc(sizeof(int) * " + jName + ".dimNum);\n"

        for i in joinAttr.factIndex:
            for j in range(0, len(factInputList)):
                if i == factInputList[j].column_name:
                    break
            factIndex.append(j)

        for i in range(0, dimNum):
            print >>fo, "\t" + jName + ".dimIndex[" + str(i) + "] = " + str(joinAttr.dimIndex[i]) + ";"
            print >>fo, "\t" + jName + ".factIndex[" + str(i) + "] = " + str(factIndex[i]) + ";"
            dimName = joinAttr.dimTables[i].table_name.lower() + "Table"
            print >>fo, "\t" + jName + ".dimTable["+str(i) + "] = " + dimName + ";\n"

        print >>fo, "\t" + jName + ".totalAttr = " + str(outputNum) + ";"
        print >>fo, "\t" + jName + ".keepInGpu = (int *) malloc(sizeof(int) * " + str(outputNum) + ");"

        if keepInGpu == 0:
            print >>fo, "\tfor(int k=0;k<" + str(outputNum)  + ";k++)"
            print >>fo, "\t\t" + jName + ".keepInGpu[k] = 0;\n"
        else:
            print >>fo, "\tfor(int k=0;k<" + str(outputNum)  + ";k++)"
            print >>fo, "\t\t" + jName + ".keepInGpu[k] = 1;\n"

        print >>fo, "\t" + jName +".attrType = (int *) (malloc(sizeof(int) * "+ jName + ".totalAttr));"
        print >>fo, "\t" + jName +".attrSize = (int *) (malloc(sizeof(int) * "+ jName + ".totalAttr));"

        tupleSize = "0"

        for i in range(0, outputNum):
            colType = outputList[i].column_type
            ctype = to_ctype(colType)
            newExp = ystree.__trace_to_leaf__(joinAttr.joinNode[dimNum-1], outputList[i], False)
            colLen = type_length(newExp.table_name,newExp.column_name,colType)
            tupleSize = tupleSize + "+" + colLen
            print >>fo, "\t" + jName + ".attrType[" + str(i) + "] = " + ctype + ";"
            print >>fo, "\t" + jName + ".attrSize[" + str(i) + "] = " + str(colLen) + ";"

        print >>fo, "\t" + jName + ".tupleSize = " + tupleSize + ";\n"

        factOutputNum = 0
        factOutputIndex = []
        dimOutputExp = []
        factOutputPos = []
        dimPos = []

        for i in range(0, outputNum):
            newExp = ystree.__trace_to_leaf__(joinAttr.joinNode[dimNum-1], outputList[i], False)
            if newExp.table_name == joinAttr.factTables[0].table_name:
                factOutputNum +=1
                for j in range(0, len(factInputList)):
                    if newExp.column_name == factInputList[j].column_name:
                        break
                factOutputIndex.append(j)
                factOutputPos.append(i)

            else:
                dimOutputExp.append(newExp)
                dimPos.append(i)


        print >>fo, "\t" + jName + ".factOutputNum = " + str(factOutputNum) + ";"
        print >>fo, "\t" + jName + ".factOutputIndex = (int *) malloc(" + jName + ".factOutputNum * sizeof(int));"
        print >>fo, "\t" + jName + ".factOutputPos = (int *) malloc(" + jName + ".factOutputNum * sizeof(int));"
        for i in range(0, factOutputNum):
            print >>fo, "\t" + jName + ".factOutputIndex[" + str(i) + "] = " + str(factOutputIndex[i]) + ";"
            print >>fo, "\t" + jName + ".factOutputPos[" + str(i) + "] = " + str(factOutputPos[i]) + ";"

        dimOutputTotal = outputNum - factOutputNum

        print >>fo, "\t" + jName + ".dimOutputTotal = " + str(dimOutputTotal) + ";"
        print >>fo, "\t" + jName + ".dimOutputNum = (int *) malloc( sizeof(int) * " + jName + ".dimNum);"
        print >>fo, "\t" + jName + ".dimOutputIndex = (int **) malloc( sizeof(int*) * " + jName + ".dimNum);"
        print >>fo, "\t" + jName + ".dimOutputPos = (int *) malloc( sizeof(int) * " + jName + ".dimOutputTotal);"

        dimOutputPos = []
        for i in range(0, len(joinAttr.dimTables)):

            dimOutputNum = len(joinAttr.outIndex[i][1])
            print >>fo, "\t" + jName + ".dimOutputNum[" + str(i) + "] = " + str(dimOutputNum) + ";"

            if dimOutputNum >0:
                print >>fo, "\t" + jName + ".dimOutputIndex[" + str(i) + "] = (int *) malloc(sizeof(int) *" +str(dimOutputNum) + ");"
                dimTableName = joinAttr.dimTables[i].table_name
                dimExp = []
                for exp in dimOutputExp:
                    if exp.table_name == dimTableName:
                        dimExp.append(exp)
                        pos = dimPos[dimOutputExp.index(exp)]
                        dimOutputPos.append(pos)

                for exp in dimExp:
                    tmpList = joinAttr.dimTables[i].select_list.tmp_exp_list
                    for j in range(0, len(tmpList)):
                        if tmpList[j].column_name == exp.column_name:
                            print >>fo, "\t" + jName + ".dimOutputIndex[" + str(i) + "][" + str(dimExp.index(exp)) + "] = " + str(j) + ";"
                            break

        for i in range(0, dimOutputTotal):
            print >>fo, "\t" + jName + ".dimOutputPos[" + str(i) + "] = " + str(dimOutputPos[i]) + ";"
                

        print >>fo, "\tint pass = " + factName + "->tupleNum / BLOCK + 1;"
        print >>fo, "\tlong tupleUnit = " + factName + "->tupleNum / pass;"
        print >>fo, "\tlong nextScan = tupleUnit;"
        print >>fo, "\tlong restTuple = " + factName + "->tupleNum;"
        print >>fo, "\tlong tupleOffset = 0;"
        print >>fo, "\toffset = 0;\n"

        print >>fo, "\tfor(int i=0;i<pass;i++){\n"
        print >>fo, "\t\tif(restTuple < nextScan)"
        print >>fo, "\t\t\tnextScan = restTuple;\n"
        print >>fo, "\t\t" + factName + "->tupleNum = nextScan;"

        for i in range(0,totalAttr):
            col = joinAttr.factTables[0].select_list.tmp_exp_list[i]
            colType = col.column_type
            colIndex =  col.column_name
            colLen = type_length(joinAttr.factTables[0].table_name, colIndex, colType)

            print >>fo, "\t\toutFd = open(\"" + joinAttr.factTables[0].table_name + str(colIndex) + "\", O_RDONLY);"
            print >>fo, "\t\tread(outFd, &header, sizeof(struct columnHeader));"
            print >>fo, "\t\t" + factName + "->dataFormat[" + str(i) + "] = header.format;"
            print >>fo, "\t\tif(header.format == UNCOMPRESSED){"
            print >>fo, "\t\t\toffset = tupleOffset *" + colLen + "+sizeof(struct columnHeader);"
            print >>fo, "\t\t\toutSize = nextScan*" + colLen + ";"
            print >>fo, "\t\t\toutTable = (char *)mmap(0,outSize+offset,PROT_READ,MAP_SHARED,outFd,0);"
            print >>fo, "\t\t\t" + factName + "->content[" + str(i) + "] = (char *) malloc(outSize);\n"
            print >>fo, "\t\t\tmemcpy(" + factName + "->content[" + str(i) + "], outTable+offset, outSize);"
            print >>fo, "\t\t\tmunmap(outTable,outSize + offset);"
            print >>fo, "\t\t\tclose(outFd);"
            print >>fo, "\t\t\t" + factName + "->attrTotalSize[" + str(i) + "] = outSize;"
            print >>fo, "\t\t}else if (header.format == DICT){"
            print >>fo, "\t\t\tstruct dictHeader dheader;"
            print >>fo, "\t\t\tread(outFd, &dheader, sizeof(struct dictHeader));"
            print >>fo, "\t\t\toffset = tupleOffset * dheader.bitNum / 8 + sizeof(struct columnHeader) + sizeof(struct dictHeader);"
            print >>fo, "\t\t\toutSize = nextScan * dheader.bitNum / 8 + sizeof(struct dictHeader);"
            print >>fo, "\t\t\toutTable = (char *)mmap(0,outSize+offset,PROT_READ,MAP_SHARED,outFd,0);"
            print >>fo, "\t\t\t" + factName + "->content[" + str(i) + "] = (char *) malloc(outSize);\n"
            print >>fo, "\t\t\tmemcpy(" + factName + "->content[" + str(i) + "], &dheader, sizeof(struct dictHeader));"
            print >>fo, "\t\t\tmemcpy(" + factName + "->content[" + str(i) + "] + sizeof(struct dictHeader), outTable+offset, outSize - sizeof(struct dictHeader));"
            print >>fo, "\t\t\tmunmap(outTable,outSize + offset);"
            print >>fo, "\t\t\tclose(outFd);"
            print >>fo, "\t\t\t" + factName + "->attrTotalSize[" + str(i) + "] = outSize;"
            print >>fo, "\t\t}else if (header.format == RLE){"
            print >>fo, "\t\t\toffset = sizeof(struct columnHeader);"
            print >>fo, "\t\t\toutSize = lseek(outFd, 0, SEEK_END) - offset;"
            print >>fo, "\t\t\toutTable = (char *)mmap(0,outSize+offset,PROT_READ,MAP_SHARED,outFd,0);"
            print >>fo, "\t\t\t" + factName + "->content[" + str(i) + "] = (char *) malloc(outSize);\n"
            print >>fo, "\t\t\tmemcpy(" + factName + "->content[" + str(i) + "], outTable+offset, outSize);"
            print >>fo, "\t\t\tmunmap(outTable,outSize + offset);"
            print >>fo, "\t\t\tclose(outFd);"
            print >>fo, "\t\t\t" + factName + "->attrTotalSize[" + str(i) + "] = outSize;"
            print >>fo, "\t\t}"

        if joinAttr.factTables[0].where_condition is not None:
            whereExp = joinAttr.factTables[0].where_condition.where_condition_exp
            whereList = []
            relList = []
            conList = []

            get_where_attr(whereExp,whereList,relList,conList)
            newWhereList = []
            whereLen = count_whereList(whereList, newWhereList)
            nested = count_whereNested(whereExp)

            if nested !=0:
                print "Not supported yet: the where expression is too complicated"
                print 1/0

            relName = joinAttr.factTables[0].table_name.lower() + "Rel"
            print >>fo, "\t\tstruct scanNode " + relName + ";"
            print >>fo, "\t\t" + relName + ".tn = " + factName + ";"
            print >>fo, "\t\t" + relName + ".hasWhere = 1;"
            print >>fo, "\t\t" + relName + ".whereAttrNum = " + str(whereLen) + ";"
            print >>fo, "\t\t" + relName + ".whereAttrType = (int *)malloc(sizeof(int)*" + str(whereLen) + ");"
            print >>fo, "\t\t" + relName + ".whereAttrSize = (int *)malloc(sizeof(int)*" + str(whereLen) + ");"
            print >>fo, "\t\t" + relName + ".whereSize = (int *)malloc(sizeof(int)*" + str(whereLen) + ");"
            print >>fo, "\t\t" + relName + ".whereIndex = (int *)malloc(sizeof(int)*" + str(whereLen) + ");"
            print >>fo, "\t\t" + relName + ".whereFormat = (int *)malloc(sizeof(int)*" + str(whereLen) + ");"
            print >>fo, "\t\t" + relName + ".wherePos = (int *)malloc(sizeof(int)*" + str(whereLen) + ");"
            print >>fo, "\t\t" + relName + ".content = (char **)malloc(sizeof(char *)*" + str(whereLen) + ");"
            if keepInGpu == 0:
                print >>fo, "\t\t" + relName + ".keepInGpu = 0;"
            else:
                print >>fo, "\t\t" + relName + ".keepInGpu = 1;"

            for i in range(0,len(newWhereList)):
                colIndex = int(newWhereList[i].column_name)
                colType = schema[newWhereList[i].table_name].column_list[colIndex].column_type
                colLen = type_length(newWhereList[i].table_name,colIndex,colType)
                ctype = to_ctype(colType)

                print >>fo, "\t\t" + relName + ".whereAttrType["+str(i) + "] = " + ctype + ";"
                print >>fo, "\t\t" + relName + ".whereAttrSize["+str(i) + "] = " + str(colLen) + ";"
                print >>fo, "\t\t" + relName + ".whereIndex["+str(i) + "] = " + str(colIndex) + ";"
                print >>fo, "\t\toutFd = open(\""+joinAttr.factTables[0].table_name+str(colIndex)+"\",O_RDONLY);"
                print >>fo, "\t\tread(outFd, &header, sizeof(struct columnHeader));"
                print >>fo, "\t\t" + relName + ".whereFormat[" + str(i) + "] = header.format;"
                print >>fo, "\t\tif(header.format == UNCOMPRESSED){"
                print >>fo, "\t\t\toffset = tupleOffset *" + colLen + " + sizeof(struct columnHeader);"
                print >>fo, "\t\t\toutSize = nextScan *" + colLen + ";"
                if UVA == 0:
                    print >>fo, "\t\t\t"+relName+".content["+str(i)+"] = (char *) malloc(outSize);"
                    print >>fo, "\t\t\t" + relName + ".wherePos[" + str(i) + "] = MEM;"
                else:
                    print >>fo, "\t\t\t" + relName + ".wherePos[" + str(i) + "] = UVA;"
                    print >>fo, "\t\t\tCUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void**)&"+relName+".content["+str(i)+"],outSize));"
                print >>fo, "\t\t\toutTable =(char *) mmap(0,outSize + offset,PROT_READ,MAP_SHARED,outFd,0);"
                print >>fo, "\t\t\tmemcpy("+relName+".content["+str(i)+"],outTable + offset,outSize);"
                print >>fo, "\t\t\tmunmap(outTable, outSize + offset);"
                print >>fo, "\t\t\tclose(outFd);"
                print >>fo, "\t\t\t"+relName+".whereSize["+str(i)+"] = outSize;"
                print >>fo, "\t\t}else if(header.format == DICT){"
                print >>fo, "\t\t\tstruct dictHeader dheader;"
                print >>fo, "\t\t\tread(outFd, &dheader, sizeof(struct dictHeader));"
                print >>fo, "\t\t\toffset = tupleOffset * dheader.bitNum / 8 + sizeof(struct columnHeader) + sizeof(struct dictHeader);"
                print >>fo, "\t\t\toutSize = nextScan * dheader.bitNum / 8 + sizeof(struct dictHeader);"
                print >>fo, "\t\t\toutTable = (char *)mmap(0,outSize+offset,PROT_READ,MAP_SHARED,outFd,0);"
                print >>fo, "\t\t\t"+relName+".content["+str(i)+"] = (char *) malloc(outSize);"
                print >>fo, "\t\t\tmemcpy("+relName+".content["+str(i)+"],&dheader, sizeof(struct dictHeader));"
                print >>fo, "\t\t\tmemcpy("+relName+".content["+str(i)+"] + sizeof(dictHeader),outTable + offset,outSize - sizeof(struct dictHeader));"
                print >>fo, "\t\t\tmunmap(outTable, outSize + offset);"
                print >>fo, "\t\t\tclose(outFd);"
                print >>fo, "\t\t\t"+relName+".whereSize["+str(i)+"] = outSize;"
                print >>fo, "\t\t}else if (header.format == RLE){"
                print >>fo, "\t\t\toffset = sizeof(struct columnHeader);"
                print >>fo, "\t\t\toutSize = lseek(outFd,0,SEEK_END) - offset;"
                print >>fo, "\t\t\toutTable = (char *)mmap(0,outSize+offset,PROT_READ,MAP_SHARED,outFd,0);"
                print >>fo, "\t\t\t"+relName+".content["+str(i)+"] = (char *) malloc(outSize);"
                print >>fo, "\t\t\toutTable =(char *) mmap(0,outSize + offset,PROT_READ,MAP_SHARED,outFd,0);"
                print >>fo, "\t\t\tmemcpy("+relName+".content["+str(i)+"],outTable + offset,outSize);"
                print >>fo, "\t\t\tmunmap(outTable, outSize + offset);"
                print >>fo, "\t\t\tclose(outFd);"
                print >>fo, "\t\t\t"+relName+".whereSize["+str(i)+"] = outSize;"
                print >>fo, "\t\t}"

            print >>fo, "\t\t" + relName + ".filter = (struct whereCondition *)malloc(sizeof(struct whereCondition));"

            print >>fo, "\t\t(" + relName + ".filter)->nested = 0;"
            print >>fo, "\t\t(" + relName + ".filter)->expNum = " + str(len(whereList)) + ";"
            print >>fo, "\t\t(" + relName + ".filter)->exp = (struct whereExp*) malloc(sizeof(struct whereExp) *" + str(len(whereList)) + ");"

            if joinAttr.factTables[0].where_condition.where_condition_exp.func_name in ["AND","OR"]:
                print >>fo, "\t\t(" + relName + ".filter)->andOr = " + joinAttr.factTables[0].where_condition.where_condition_exp.func_name + ";"

            else:
                print >>fo, "\t\t(" + relName + ".filter)->andOr = EXP;"

            for i in range(0,len(whereList)):
                colIndex = -1
                for j in range(0,len(newWhereList)):
                    if newWhereList[j].compare(whereList[i]) is True:
                        colIndex = j
                        break

                if colIndex <0:
                    print 1/0

                print >>fo, "\t\t(" + relName + ".filter)->exp[" + str(i) + "].index = " + str(colIndex) + ";"
                print >>fo, "\t\t(" + relName + ".filter)->exp[" + str(i) + "].relation = " + relList[i] + ";" 

                colType = whereList[i].column_type
                ctype = to_ctype(colType)

                if ctype == "INT":
                    print >>fo, "\t\t{"
                    print >>fo, "\t\t\tint tmp = " + conList[i] + ";"
                    print >>fo, "\t\t\tmemcpy((" + relName + ".filter)->exp[" + str(i) + "].content, &tmp,sizeof(int));"
                    print >>fo, "\t\t}"

                elif ctype == "FLOAT":

                    print >>fo, "\t\t{"
                    print >>fo, "\t\t\tfloat tmp = " + conList[i] + ";"
                    print >>fo, "\t\t\tmemcpy((" + relName + ".filter)->exp[" + str(i) + "].content, &tmp,sizeof(float));"
                    print >>fo, "\t\t}"
                    print 1/0
                else:
                    print >>fo, "\t\tstrcpy((" + relName + ".filter)->exp[" + str(i) + "].content," + conList[i] + ");\n"

            print >>fo, "\t\ttableScan(&" + relName + ", &pp);"
            print >>fo, "\t\tfreeScan(&" + relName + ");"

        print >>fo, "\t\tstruct tableNode *join1 = hashJoin(&" + jName + ", &pp);"
        for i in range(0, totalAttr):
                print >>fo, "\t\tif(" + factName + "->dataPos[" + str(i) + "] == MEM)"
                print >>fo, "\t\t\tfree(" + factName + "->content[" + str(i) + "]);"
                print >>fo, "\t\telse"
                print >>fo, "\t\t\tcudaFree(" + factName + "->content[" + str(i) + "]);"

        print >>fo, "\t\tif(pass !=1){"
        print >>fo, "\t\t\tmergeIntoTable(" + resultNode + ",join1,&pp);"
        print >>fo, "\t\t\tfreeTable(join1);"
        print >>fo, "\t\t}else"
        print >>fo, "\t\t\t" + resultNode + "=join1;"
        print >>fo, "\t\ttupleOffset +=tupleUnit;"
        print >>fo, "\t\trestTuple -= nextScan;"

        print >>fo, "\t}\n"

    ### the type 1 join ends ###

############# facttable ends ###################

    if len(aggNode) >0 :
        gb_exp_list = aggNode[0].group_by_clause.groupby_exp_list
        select_list = aggNode[0].select_list.tmp_exp_list
        selectLen = len(select_list)
        gbLen = len(gb_exp_list)
        print >>fo, "\tstruct groupByNode * gbNode = (struct groupByNode *) malloc(sizeof(struct groupByNode));"
        print >>fo, "\tgbNode->table = " +resultNode +";"
        print >>fo, "\tgbNode->groupByColNum = " + str(gbLen) + ";"
        print >>fo, "\tgbNode->groupByIndex = (int *)malloc(sizeof(int) * " + str(gbLen) + ");"
        print >>fo, "\tgbNode->groupByType = (int *)malloc(sizeof(int) * " + str(gbLen) + ");"
        print >>fo, "\tgbNode->groupBySize = (int *)malloc(sizeof(int) * " + str(gbLen) + ");"

        for i in range(0,gbLen):
            exp = gb_exp_list[i]
            if isinstance(exp, ystree.YRawColExp):
                print >>fo, "\tgbNode->groupByIndex[" + str(i) + "] = " + str(exp.column_name) + ";"
                print >>fo, "\tgbNode->groupByType[" + str(i) + "] = gbNode->table->attrType[" + str(exp.column_name) + "];" 
                print >>fo, "\tgbNode->groupBySize[" + str(i) + "] = gbNode->table->attrSize[" + str(exp.column_name) + "];" 
            elif isinstance(exp, ystree.YConsExp):
                print >>fo, "\tgbNode->groupByIndex[" + str(i) + "] = -1;" 
                print >>fo, "\tgbNode->groupByType[" + str(i) + "] = INT;" 
                print >>fo, "\tgbNode->groupBySize[" + str(i) + "] = sizeof(int);" 
            else:
                print 1/0

        print >>fo, "\tgbNode->outputAttrNum = " + str(selectLen) + ";"
        print >>fo, "\tgbNode->attrType = (int *) malloc(sizeof(int) *" + str(selectLen) + ");"
        print >>fo, "\tgbNode->attrSize = (int *) malloc(sizeof(int) *" + str(selectLen) + ");"
        print >>fo, "\tgbNode->tupleSize = 0;"
        print >>fo, "\tgbNode->gbExp = (struct groupByExp *) malloc(sizeof(struct groupByExp) * " + str(selectLen) + ");"

        for i in range(0,selectLen):
            exp = select_list[i]
            if isinstance(exp, ystree.YFuncExp):

                print >>fo, "\tgbNode->tupleSize += sizeof(float);"
                print >>fo, "\tgbNode->attrType[" + str(i) + "] = FLOAT;"
                print >>fo, "\tgbNode->attrSize[" + str(i) + "] = sizeof(float);"
                print >>fo, "\tgbNode->gbExp["+str(i)+"].func = " + exp.func_name + ";"
                para = exp.parameter_list[0]
                mathFunc = mathExp()
                mathFunc.addOp(para)
                prefix = "\tgbNode->gbExp[" + str(i) + "].exp"
                printMathFunc(fo,prefix, mathFunc)

            elif isinstance(exp, ystree.YRawColExp):
                colIndex = exp.column_name
                print >>fo, "\tgbNode->attrType[" + str(i) + "] = " + resultNode + "->attrType[" + str(colIndex) + "];"
                print >>fo, "\tgbNode->attrSize[" + str(i) + "] = " + resultNode + "->attrSize[" + str(colIndex) + "];"
                print >>fo, "\tgbNode->tupleSize += "+resultNode + "->attrSize[" + str(colIndex) + "];"
                print >>fo, "\tgbNode->gbExp[" + str(i) + "].func = NOOP;"
                print >>fo, "\tgbNode->gbExp[" + str(i) + "].exp.op = NOOP;"
                print >>fo, "\tgbNode->gbExp[" + str(i) + "].exp.exp = NULL;"
                print >>fo, "\tgbNode->gbExp[" + str(i) + "].exp.opNum = 1;"
                print >>fo, "\tgbNode->gbExp[" + str(i) + "].exp.opType = COLUMN;"
                print >>fo, "\tgbNode->gbExp[" + str(i) + "].exp.opValue = " + str(exp.column_name) + ";"

            else:
                if exp.cons_type == "INTEGER":
                    print >>fo, "\tgbNode->attrType[" + str(i) + "] = INT;"
                    print >>fo, "\tgbNode->attrSize[" + str(i) + "] = sizeof(int);"
                    print >>fo, "\tgbNode->tupleSize += sizeof(int);"
                elif exp.cons_type == "FLOAT":
                    print >>fo, "\tgbNode->attrType[" + str(i) + "] = FLOAT;"
                    print >>fo, "\tgbNode->attrSize[" + str(i) + "] = sizeof(float);"
                    print >>fo, "\tgbNode->tupleSize += sizeof(float);"
                else:
                    print 1/0

                print >>fo, "\tgbNode->gbExp[" + str(i) + "].func = NOOP;"
                print >>fo, "\tgbNode->gbExp[" + str(i) + "].exp.op = NOOP;"
                print >>fo, "\tgbNode->gbExp[" + str(i) + "].exp.exp = NULL;"
                print >>fo, "\tgbNode->gbExp[" + str(i) + "].exp.opNum = 1;"
                print >>fo, "\tgbNode->gbExp[" + str(i) + "].exp.opType = CONS;"
                print >>fo, "\tgbNode->gbExp[" + str(i) + "].exp.opValue = " + str(exp.cons_value) + ";"

        resultNode = "gbResult"
        print >>fo, "\tstruct tableNode * " + resultNode + " = groupBy(gbNode, &pp);"
        print >>fo, "\tfreeGroupByNode(gbNode);\n"

    if len(orderbyNode) > 0 :
        orderby_exp_list = orderbyNode[0].order_by_clause.orderby_exp_list
        odLen = len(orderby_exp_list)
        print >>fo, "\tstruct orderByNode * odNode = (struct orderByNode *) malloc(sizeof(struct orderByNode));"
        print >>fo, "\todNode->table = " +resultNode +";"
        print >>fo, "\todNode->orderByNum = " + str(odLen) + ";"
        print >>fo, "\todNode->orderBySeq = (int *) malloc(sizeof(int) * odNode->orderByNum);"
        print >>fo, "\todNode->orderByIndex = (int *) malloc(sizeof(int) * odNode->orderByNum);"

        for i in range(0,odLen):
            seq = orderbyNode[0].order_by_clause.order_indicator_list[i]
            if seq == "ASC":
                print >>fo, "\todNode->orderBySeq[" + str(i) + "] = ASC;"
            else:
                print >>fo, "\todNode->orderBySeq[" + str(i) + "] = DESC;"

            print >>fo, "\todNode->orderByIndex[" + str(i) + "] = " + str(orderby_exp_list[i].column_name) + ";"

        resultNode = "odResult"
        print >>fo, "\tstruct tableNode * " + resultNode + " = orderBy(odNode,&pp);"
        print >>fo, "\tfreeOrderByNode(odNode);\n"

    print >>fo, "\tstruct materializeNode mn;"
    print >>fo, "\tmn.table = "+resultNode + ";"
    print >>fo, "\tmaterializeCol(&mn, &pp);"
    print >>fo, "\tfreeTable("+resultNode + ");\n"
    print >>fo, "}\n"

    fo.close()

def ysmart_code_gen(argv):
    pwd = os.getcwd()
    resultdir = "./src"
    codedir = "./GPUCODE"
    schemaFile = None 

    if len(sys.argv) == 3:
        tree_node = ystree.ysmart_tree_gen(argv[1],argv[2])

    elif len(sys.argv) == 2:
        schemaFile = ystree.ysmart_get_schema(argv[1])

    if len(sys.argv) == 3 and tree_node is None:
        exit(-1)

    if os.path.exists(resultdir) is False:
        os.makedirs(resultdir)

    os.chdir(resultdir)
    if os.path.exists(codedir) is False:
        os.makedirs(codedir)

    os.chdir(codedir)

    generate_schema_file()
    generate_loader()

    if len(sys.argv) == 3:
        generate_code(tree_node)

    if schemaFile is not None:
        metaFile = open(".metadata",'wb')
        pickle.dump(schemaFile, metaFile)
        metaFile.close()

    os.chdir(pwd)

