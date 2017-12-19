#!/usr/bin/python
# encoding=utf-8
import os
import re
import json
import sys
#reload(sys)
#sys.setdefaultencoding( "utf-8" )
fileMap = {}
result = open('/Users/edz/PycharmProjects/myAIProject/txtClassifier.txt', 'a')

#获取集合
def getFileName(rootdir):
    #文件下所有文件
    fileNames = os.listdir(rootdir)
    #遍历所有文件
    for filename in fileNames:
        #拼接文件名
        allname = rootdir+'/'+filename
        #如果是文件夹
        if os.path.isdir(allname):
            #递归调用本方法
            getFileName(allname)
        #如果是文件
        else:
            roots=rootdir.split('/')
            root=roots[6:]
            rootstr=''
            for a in root:
                rootstr +="/"+a
            #如果有值
            if fileMap.has_key(filename):
                filename=filename.__str__()
                fileMap[filename]=fileMap[filename]+[rootstr]
            #如果为空
            else:
                fileMap[filename]=[rootstr]
    return fileMap
#排序并打印集合
def printMap(fileMap):
    sortedMap = sorted(fileMap.iteritems(),key=lambda asd:asd[1].__len__(),reverse=True)
    for i in sortedMap:
        s1 = i[0]+":"+str(i[1])
        print s1
        result.write(s1+"""
""")
        pass
    pass

f1=getFileName('/Users/edz/PycharmProjects/myAIProject/speech_corpus')
#f1={'a':[1,2,3],'b':[2,2]}
printMap(f1)

