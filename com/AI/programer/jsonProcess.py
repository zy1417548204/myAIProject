#!/usr/bin/python
# encoding=utf-8
import os
import re
import json
import sys

"""
json处理，每个json文本输出一个txt文本，挑出开始时间和内容
"""
#定义最终输出脚本
#result = open('/Users/edz/PycharmProjects/audio_recognition_project/Ncresult.txt', 'a')
reload(sys)
sys.setdefaultencoding( "utf-8" )
#生成文件夹
os.mkdir("/Users/edz/PycharmProjects/myAIProject/jsonPro")
def detectnc(path,result):
    jfile = open(path)
    hjson = json.load(jfile)
    for a in hjson:
        s2 = (str(a["bg"]) + ":" + str(a["ed"]) + ":" + str(a["onebest"]))
        print s2
        result.write(s2 +"""
""")
        pass
    pass

 # 加载文件
def loadAndoutput():
    txtNames = os.listdir("/Users/edz/PycharmProjects/myAIProject/recognition_result")
    for txtName in txtNames:
        result = open('/Users/edz/PycharmProjects/myAIProject/'+txtName,'a')
        # print txtName
        detectnc("/Users/edz/PycharmProjects/myAIProject/recognition_result/"+txtName,result)
    pass

loadAndoutput()