#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import json
import sys

reload(sys)
sys.setdefaultencoding( "utf-8" )
"""
将错误的json属性替换成成正确的并输出
1。读取json文本，如没有错，就原样输出
2。如有问题
"""

#读取文本数组
#得到{合格-20052769陈建明.txt:{9280:什么事你说对吗,555650:哪儿上}}
def ectractIofo(txtfile):
    file = open(txtfile)
    map = {}
    while True:
        line = file.readline()
        attrbutes=line.split(':')
        attrbute =attrbutes[0].split('/')
        print attrbutes.__len__()
        if attrbutes.__len__()<2:
            break
        # 如果有值
        if map.has_key(str(attrbute[6])):
            map[attrbute[6]][attrbutes[1]]=attrbutes[5]
        # 如果为空
        else:

            map[str(attrbute[6])]={attrbutes[1]:attrbutes[5]}


    return map
    pass  # do something




 # (主方法)加载文件，参数path是要修正的json文件夹,txt是要读取的文本，根据这个文本来修改
def loadAndoutput(path,txt):
    os.mkdir("/Users/edz/PycharmProjects/myAIProject/jsonModify")
    txtNames = os.listdir(path)
    jmap = ectractIofo(txt)
    print jmap.__len__()
    #遍历文件下所有文本
    for txtName in txtNames:
        result = open('/Users/edz/PycharmProjects/myAIProject/jsonModify/'+txtName,'a')

        #如果这个文本有错误的json
        if jmap.has_key(str(txtName)):
            #打开json文件
            jfile = open('/Users/edz/PycharmProjects/myAIProject/extract_json/'+txtName)
            hjson = json.load(jfile)
            jarr = []
            #遍历每个json
            for a in hjson:
                #json有错修正并输出
                if jmap[txtName].has_key(str(a["start_time"])):
                    print jmap[txtName][a["start_time"]]
                    #将错误内容修改过来
                    a["onebest"]=jmap[txtName][a["start_time"]].__str__().decode('utf-8')
                    print a["onebest"]
                    jarr=jarr+[a]
                    print
                #json没错，输出出来
                else:
                    jarr =jarr+[a]
                    pass
            #将jarr数组打印出来
            detectnc(jarr, result)
            pass
        #这个文本没有错，原样输出来
        else:
            jfile = open('/Users/edz/PycharmProjects/myAIProject/extract_json/'+txtName)
            hjson = json.load(jfile)
            detectnc(hjson,result)
    pass

# jfile = open("/Users/edz/PycharmProjects/myAIProject/recognition_result/2王业20121514_json.txt")
# hjson = json.load(jfile)

#替换掉u'
def remove_uni(s):
    """remove the leading unicode designator from a string"""
    s2 = ''
    if s.__contains__("u'"):
        s2 = s.replace("u'", "'")
    elif s.__contains__('u"'):
        s2 = s.replace('u"', '"')
    return s2




#os.mkdir("/Users/edz/PycharmProjects/myAIProject/jsonModify")
#将数组输出到某个文件
def detectnc(arr,result):
    s2=remove_uni(str(arr).decode('unicode-escape').encode('utf-8'))
    result.write(s2)
    pass


#print os.readlink('/Users/edz/PycharmProjects/myAIProject/new_发张阳.txt')
#开始调用
txt='/Users/edz/PycharmProjects/myAIProject/new-发张阳.txt'
path='/Users/edz/PycharmProjects/myAIProject/extract_json'
loadAndoutput(path,txt)







# for a in hjson:
#     #unicode(‘哈’, ‘utf-8′).decode(‘utf-8′)
#      print str(a).encode('ascii')