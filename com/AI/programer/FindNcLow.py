#!/usr/bin/python
# encoding=utf-8
import os
import re
import json
import sys
"""
检测nc小于一的并将起列举出来
"""
#定义最终输出脚本
result = open('/Users/edz/PycharmProjects/audio_recognition_project/Ncresult.txt', 'a')
reload(sys)
sys.setdefaultencoding( "utf-8" )
def detectnc(path):
    jfile = open(path)
    hjson = json.load(jfile)
    for a in hjson:
        if a["nc"]<1:
            s2 = ( path + ":" + str(a["bg"]) + ":" + str(a["ed"]) + ":" + str(a["nc"]) + ":" + str(a["onebest"]))
            print s2
            result.write(s2 +"""
""")
        else:
            print "ni"
            pass
        pass
    pass

 # 加载文件
def loadAndoutput():
    txtNames = os.listdir("/Users/edz/PycharmProjects/myAIProject/recognition_result")
    for txtName in txtNames:
        # print txtName
        detectnc("/Users/edz/PycharmProjects/myAIProject/recognition_result/"+txtName)
    pass

loadAndoutput()