#! /usr/bin/python

import sys
import os
import shutil

def dbHelp():
    print "Command:"
    print "\tcreate DBName: create the database"
    print "\tdelete DBName: delete the database"
    print "\tlist: list the table infomation in the database"
    print "\ttranslate SQL: translate SQL into CUDA file"
    print "\texecute SQL: translate and execute given SQL on GPU"
    print "\tload TableName data: load data into the given table"
    print "\texit"

def dbCreate(dbName):

    ret = 0
    dbTop = "database"

    if not os.path.exists(dbTop):
        os.makedirs(dbTop)

    dbPath = dbTop + "/" + dbName

    if os.path.exists(dbPath):
        return -1

    os.makedirs(dbPath)

    cmd = 'python XML2MapReduce/main.py ' + schemaFile
    ret = os.system(cmd)

    if ret !=0 :
        exit(-1)

    cmd = 'make -C src/GPUCODE/ loader &> /dev/null'
    ret = os.system(cmd)

    if ret != 0:
        exit(-1)

    cmd = 'cp src/GPUCODE/gpuDBLoader ' + dbPath
    ret = os.system(cmd)

    if ret != 0:
        exit(-1)

    return 0

def dbDelete(dbName):

    dbTop = "database"

    dbPath = dbTop + "/" + dbName
    if os.path.exists(dbPath):
        shutil.rmtree(dbPath)

if len(sys.argv) != 2:
    print "./gpudb.py schemaFile"
    exit(-1)

schemaFile = sys.argv[1]

while 1:
    ret = 0
    dbCreated = 0
    dbName = ""

    cmd = raw_input(">")
    cmdA = cmd.lstrip().rstrip().split()

    if len(cmdA) == 0:
        continue

    if cmdA[0].upper() == "HELP":
        dbHelp()

    elif cmdA[0].upper() == "?":
        dbHelp()

    elif cmdA[0].upper() == "EXIT":
        break

    elif cmdA[0].upper() == "CREATE":

        if dbCreated !=0:
            print "Already created database. Delete first."
            continue


        if len(cmdA) !=2:
            print "usage: create DBName"

        else:
            ret = dbCreate(cmdA[1].upper())
            if ret == -1:
                print cmdA[1] + " already exists"
            else:
                dbCreated = 1
                dbName = cmdA[1].upper()
                print cmdA[1] + " has been successfully created"
                


    elif cmdA[0].upper() == "DELETE":
        if len(cmdA) !=2:
            print "usage: delete DBName"

        dbCreated = 0
        dbDelete(cmdA[1].upper())

    else:
        print "Unknown command"

os.system("clear")

