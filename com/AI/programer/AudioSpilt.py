#!/usr/bin/python
# -*- coding: utf-8 -*-
import librosa
import librosa.display
import wave
#加载数据
# import soundfile as sf
# x,Fs = librosa.load("01.mp3")
# print x
# print Fs
#wavfile = open("02.mp3",'w')
#sf.write("3.mp3",Fs,x[int(17):int(27)])
#x.write(strOut, Fs, x[int(Fs * s[0]):int(Fs * s[1])])
#for  s in enumerate(newseg):
#    strOut = "{0:s}/{1:.3f}-{2:.3f}.wav".format("", s[0], s[1])
#    wavfile.write(strOut, Fs, x[int(Fs * s[0]):int(Fs * s[1])])
##################################################################################


import os as o
from pydub import AudioSegment
#测试读取json文件
import json
import Queue
#
def traverse_file(path):
    folder_list = []
    file_name_set = set()
    file_path_list = []
    for file_name in o.listdir(path):
        file_path = o.path.join(path,file_name)
        if o.path.exists(file_path):
            if o.path.isdir(file_path):
                folder_list.append(str(file_path))
                # print file_path
            elif o.path.isfile(file_path) and not file_name.endswith('DS_Store') and not file_name.endswith('.mp3'):
                if not file_name in file_name_set:
                    # print file_path
                    file_name_set.add(file_name)
                    file_path_list.append(file_path)

        while len(folder_list) > 0 :
            file_path = folder_list.pop(0)
            if o.path.exists(file_path) and o.path.isdir(file_path):
                for sub_file in o.listdir(file_path):
                    sub_file_path = o.path.join(file_path,sub_file)
                    if o.path.exists(sub_file_path) and o.path.isdir(sub_file_path):
                        folder_list.append(str(sub_file_path))
                    elif o.path.exists(sub_file_path) and o.path.isfile(sub_file_path) and sub_file.endswith('.mp3'):
                        if not sub_file in file_name_set:
                            # print sub_file_path
                            file_name_set.add(sub_file)
                            file_path_list.append(str(sub_file_path))

    file_name_list = list(file_name_set)
    return file_path_list,file_name_list

#本方法是通过解析Json文本来切割视频
def split(file_path,text_file_path,text_file_name):
    #加载mp3文件
    file_name = str.replace(text_file_name,'.txt','')
    print file_name
    song = AudioSegment.from_file(file_path,'mp3')
    o.mkdir("data/" + file_name)
    #加载json文本
    jfile = open(text_file_path)
    hjson = json.load(jfile)
    for text_seg in hjson:
        start_time = str(text_seg["start_time"])
        end_time = str(text_seg["end_time"])
        audio_piece = song[int(start_time) : int(end_time)]
        audio_piece.export("data/" + file_name + '/' + start_time + "~" + end_time + ".mp3", format="mp3")

python_path_index = o.getcwd().index('src/main/python/audio')
source_path = str(o.getcwd())[0:python_path_index] + 'source/extract_json'

file_path_list, file_name_list = traverse_file('/Users/administrator/Documents/speech_corpus')
text_name_list = o.listdir(source_path)

size = len(file_path_list)

for i in range(size):
    audio_file_path = file_path_list[i]
    audio_file_name = file_name_list[i]
    text_file_name = str.replace(audio_file_name,'.mp3','.txt')
    text_file_path = str(o.path.join(source_path + '/' + text_file_name))
    if str.replace(audio_file_name,'.mp3','.txt') in text_name_list:
        split(audio_file_path,text_file_path,text_file_name)


# Beat tracking example
# from __future__ import print_function
# import librosa
#
# # 1. Get the file path to the included audio example
# filename = librosa.util.example_audio_file("01.mp3")
#
# # 2. Load the audio as a waveform `y`
# #    Store the sampling rate as `sr`
# y, sr = librosa.load(filename)
#
# # 3. Run the default beat tracker
# tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#
# print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
#
# # 4. Convert the frame indices of beat events into timestamps
# beat_times = librosa.frames_to_time(beat_frames, sr=sr)
#
# print('Saving output to beat_times.csv')
# librosa.output.times_csv('beat_times.csv', beat_times)