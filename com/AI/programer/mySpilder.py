#!/usr/bin/python
# encoding=utf-8
from bs4 import BeautifulSoup as bs
import re
import urllib2 as ur
import urllib
import os as o
import requests

url1 = 'http://www.pthxx.com/b_audio/01_langdu_1/01.html'
url2 = 'http://www.pthxx.com/b_audio/pthxx_com_mp3/01_langdu/01.mp3'

#下载所有mp3文件
def downAllMp3():
    for a in range(1,61):
        if a < 10 :
            urllib.urlretrieve(url2[0:-6]+"0"+a.__str__()+".mp3","data/"+"0"+a.__str__()+"/"+"0"+a.__str__()+".mp3")
        else:
            urllib.urlretrieve(url2[0:-6]+a.__str__()+".mp3","data/"+a.__str__()+"/"+a.__str__()+".mp3")

#先下载文本在下载mp3oo
def downAllTxt():
    for a in range(1, 61):
        if a < 10:
            s = url1[0:-7]+"0"+a.__str__()+".html"
            dowloadTxt(s)
        else:
            s = url1[0:-7]  + a.__str__() + ".html"
            dowloadTxt(s)
#os.mkdir

#通过URL下载文本内容
def dowloadTxt(str1):
    o.makedirs("data/" + str1[-7:-5])
    #dowloadMp3(str1)
    html = ur.urlopen(str1).read()
    soup = bs(html)
    #保存文字
    it = soup.select('#main2')
    str2 = ''
    for a in it:
        str2 += a.get_text() + "" \
                               ""
    str3 = str2.encode("utf-8")
    save('data/'+str1[-7:-5]+'/'+str1[-7:-5]+'.txt', str3)
    #保存拼音
    ip = soup.select('#main3')
    str4 = ''
    for a in ip:
        str4 += a.get_text() + "" \
                               ""
    str5 = str4.encode("utf-8")
    save('data/'+str1[-7:-5]+'/'+str1[-7:-5]+'_py.txt', str5)
#通过URL下载mp3
def dowloadMp3(str1):
    #file1 = os.open(str[-7:-5])
    str2 = str1[0:-5]+".mp3"
    #makedirs和mkdir 坑爹啊
    #r = requests.get(str2)
    # with open("data/" + str1[-7:-5]+"/"+str1[-7:-5]+".mp3") as code :
    #     code.write()
    urllib.urlretrieve(str2, "data/" + str1[-7:-5]+"/"+str1[-7:-5]+".mp3")

def save(filename, contents):
  fh = open(filename, 'w')
  fh.write(contents)
  fh.close()
downAllTxt()
downAllMp3()