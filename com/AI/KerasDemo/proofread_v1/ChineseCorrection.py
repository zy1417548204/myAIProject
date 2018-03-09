import re
from collections import Counter

f = open(r'word_freq.txt', encoding='utf8')
WORDS = {}
id = 0
for line in f.readlines():
    if id % 2 == 0:
        word_freq = line.split('\t')
        WORDS[word_freq[0]] = int(word_freq[1])
    id = id + 1
letters = open(r'word.txt', encoding='utf8').read()

N = sum(WORDS.values())

def P(word):
    return WORDS[word] / N

def know(words):
    return set(w for w in words if w in WORDS)

def edits1(word):
    #letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)] #切分
    deletes = [L + R[1:] for L, R in splits if R] #删除
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1] #移位
    replaces = [L + c + R[1:] for L, R in splits for c in letters] #代替
    inserts = [L + c + R for L, R in splits for c in letters] #插入
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def candidates(word):
    return (know([word]) or know(edits1(word)) or know(edits2(word)) or [word])

def correction(word):
    return max(candidates(word), key=P)

print(correction('正分夺秒'))
print(correction('灿烂夺木'))
print(correction('大好清春'))
#print(correction('十里淘花'))
print(correction('西胡'))
print(correction('21'))