# -*- coding = utf-8 -*-
# @Time ： 2021/11/7 13:58
# @Author : 朱豪
# @File : data_processing.py
# @Software : PyCharm


import numpy as np
import pandas as pd
import re
from tqdm import tqdm

docs = []
with open ('NCPPolicies_context_20200301.csv', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip().split('\t', 1)
        docs.append(line)

docs = pd.DataFrame(docs[1:])
docs.columns=['docid','text']
docs_new = docs.groupby('text', as_index=False).agg({'docid': lambda x: ','.join(x).split(',')})
docs_new['new_docid'] = docs_new['docid'].apply(lambda x:x[0])


old2new = {}
useless = []
for old, new in docs_new[['docid', 'new_docid']].values:
    for i in old:
        old2new[i] = new
        if i != new:
            useless.append(i)

train = pd.read_csv('../temp_data/NCPPolicies_train_20200301.csv',delimiter='\t')
train['docid'] = train['docid'].apply(lambda x:old2new[x])

docs_new['docid'] = docs_new['new_docid']
docs_new[['docid','text']].to_csv('../temp_data/context.csv',index=False)
train.to_csv('../temp_data/train.csv', index=False)



def check_split(boundary, offset):
    for a, b in boundary:
        for i in range(len(offset)):
            if offset[i] > a and offset[i] < b:  # 夹在中间
                offset.pop(i)  # 删掉这个分割
                return offset, False
    else:
        return offset, True

def split_sentences(text, boundary):
    sentences = text.split('。')
    sentences = [s + "。" for s in sentences]
    offset = []
    res = []
    start = 0
    for s in sentences:
        offset.append(start)
        start += len(s)
    offset.append(start)
    check = False
    while not check:
        offset, check = check_split(boundary, offset)
    for i in range(len(offset) - 1):
        res.append(text[offset[i]:offset[i + 1]])
    return res


def clear_whitespace(str):
    return ' '.join(str.split())

class Solution:  # KMP字符串匹配
    # 获取next数组
    def get_next(self, T):
        i = 0
        j = -1
        next_val = [-1] * len(T)
        while i < len(T) - 1:
            if j == -1 or T[i] == T[j]:
                i += 1
                j += 1
                # next_val[i] = j
                if i < len(T) and T[i] != T[j]:
                    next_val[i] = j
                else:
                    next_val[i] = next_val[j]
            else:
                j = next_val[j]
        return next_val

    # KMP算法
    def kmp(self, S, T):
        i = 0
        j = 0
        next = self.get_next(T)
        while i < len(S) and j < len(T):
            if j == -1 or S[i] == T[j]:
                i += 1
                j += 1
            else:
                j = next[j]
        if j == len(T):
            return i - j
        else:
            return -1


para_num = {}  # 每个doc被切分为多少段落
answers = {}
def get_paragraph(target, id, text, my_dict):  # 输入一行
    boundaries = []
    s = Solution()
    flag_do_not_cut = False
    if id in answers:
        for answer in answers[id]:
            answer = clear_whitespace(answer)
            text = clear_whitespace(text)
            # print(answer)
            start = s.kmp(text, answer)
            if (start == -1):
                flag_do_not_cut = True
            else:
                boundaries.append((start, start + len(answer)))
    if flag_do_not_cut:
        paragraphs = [text]
    else:
        sentences = split_sentences(text, boundaries)
        front = 0
        threshold = 400
        paragraphs = []
        temp = ""
        for i in range(len(sentences)):
            if (len(temp) + len(sentences[i]) < threshold):
                temp += sentences[i]
            else:
                paragraphs.append(str(temp))
                temp = sentences[i]
        if (len(temp) > 0):
            paragraphs.append(str(temp))
    paragraphs = [p for p in paragraphs if len(p) > 0]
    for i in range(len(paragraphs)):
        target = target.append([{'docid': id, 'paraid': str(id) + str(i).rjust(3, '0'), 'text': paragraphs[i]}],
                               ignore_index=True)
        my_dict[str(id) + str(i).rjust(3, '0')] = paragraphs[i]
    return target, my_dict


context = []
df_train = pd.read_csv('../temp_data/train.csv')  # id,docid,question,answer
for index, row in df_train.iterrows():
    if row['docid'] not in answers:
        answers[row['docid']] = []
    answers[row['docid']].append(row['answer'])

df_context = pd.read_csv('/context.csv')
df_context_new = pd.DataFrame(columns=['docid', 'paraid', 'text'])
doc_context = {}
context = {}  # paraid_text
for index, row in tqdm(df_context.iterrows(), total=len(df_context)):
    docid = row['docid'].strip()
    text = row['text'].strip()
    doc_context[docid] = text
    df_context_new, context = get_paragraph(df_context_new, docid, text, context)
df_context_new.to_csv('../temp_data/context_fin.csv', encoding='utf-8', index=False)

cnt = 0
questions = []
problems = []
for index, row in df_train.iterrows():
    flag = False
    for i in range(para_num[row['docid']]):
        if not flag:
            paraid = str(row['docid']) + str(i).rjust(3, '0')
            s = Solution()
            start = s.kmp(clear_whitespace(context[paraid]), clear_whitespace(row['answer']))
            if start != -1:
                df_train.loc[index, 'paraid'] = paraid
                flag = True
    else:
        if not flag:
            cnt += 1
            questions.append(row['id'])
            problems.append(row['docid'])
df_train.to_csv('./temp_data/train_fin.csv', encoding='utf-8', index=False)