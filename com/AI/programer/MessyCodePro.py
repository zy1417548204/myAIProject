#!/usr/bin/python
# encoding=utf-8
import os
import re
import json
"""
#检测jonson 文件下有乱码
碰到的坑
（1）由于是git分支，打开文件用绝对路径，用相对路径报找不到文件
（2）json解析后是Ascii码，字符串拼接得强转成str,否则报类型异常
（3）python 的换行符果然另类／r/n 不管用的
"""
class TxtDetect(object):
    p2 = r"^[\u4e00-\u9fa5]"
    pattern2 = re.compile(p2)
    result = open('/Users/edz/PycharmProjects/audio_recognition_project/result.txt', 'a')
    def txtDetect(self,josonfile):
        # 加载json文本
        jfile = open(josonfile)
        hjson = json.load(jfile)
        for a in hjson:
            s1 = self.pattern2.findall(a["onebest"])
            if s1 != []:
                print josonfile
                print a["start_time"]
                s2 = (josonfile + ":" + str(a["start_time"]) + "~" + str(a["end_time"]))
                print s2
                self.result.write("""
                """ + s2)

    # 加载文件
    def loadAndoutput(self):
        txtNames = os.listdir("/Users/edz/PycharmProjects/audio_recognition_project/extract_json")
        for txtName in txtNames:
            # print txtName
            self.txtDetect("/Users/edz/PycharmProjects/audio_recognition_project/extract_json/" + txtName)
        pass


#创建实例调用loadAndoutput方法
c1 = TxtDetect
c1.loadAndoutput()
