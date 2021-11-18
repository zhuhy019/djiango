import datetime
import os
import time

from rest_framework.views import APIView
from rest_framework.response import Response
from poll.serializer import *

import jieba
import pandas as pd
import re
import numpy as np
import jieba_fast as jieba
from tqdm import tqdm
import tokenizers
import torch.nn as nn
from transformers import BertModel
import pickle
# from . import predict
from .models import *
import torch
# Create your views here.
class QuestionApiView(APIView):
    serializer_class= QuestionSerializer
    def get(self,request):
        allQuestions=Question.objects.all().values()
        return Response({"Message":"List of Questions", "Questions_List":allQuestions})
    def post(self,request):
        pred,candID = predicting(request.data.get("Question"))
        id,text1=re_text(candID)
        text =text1.values[0]
        dict = {"Question":request.data.get("Question"),'pred': pred,'text':text}
        serializer_obj = QuestionSerializer(data=dict)
        print(datetime.datetime.now())
        if(serializer_obj.is_valid()):
            Question.objects.create(
                id="".join(str(uuid.uuid4()).split("-")).upper(),
                docid=id,
                question= serializer_obj.data.get("Question"),
                pred =serializer_obj.data.get("pred"),
                content_text =serializer_obj.data.get("text"),
                createtime=datetime.datetime.now())
        return Response({"Message":"New question Added!", "Answer":text1.values[0]})

class config:
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 64
    EPOCHS = 10
    module_dir = os.path.dirname(__file__)
    BERTPATH = module_dir
    Config = os.path.join(module_dir, '../../djangoProject1/poll/config.json')
    Tokenizer = os.path.join(module_dir, '../../djangoProject1/poll/tokenizer.json')
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, 'vocab.txt')
    TOKENIZER = tokenizers.BertWordPieceTokenizer(
        file_path,
       lowercase=True
    )
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERTPATH,return_dict=False)
        self.drop_out = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 2)
        self.fc2 = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, ids, mask, token_type_ids):
        last_hidden_state, pooler_output = self.bert(
            ids,
            attention_mask=mask,
        )

        out = self.drop_out(last_hidden_state)
        logits = self.fc1(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        y = self.fc2(pooler_output)

        return start_logits, end_logits, y

def clean(doc, ngram=True):
    remove_chars = '[·’!"\#$%&\'()＃！（）*+,-.。/:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'

    stopwords = []
    module_dir = os.path.dirname(__file__)
    stopword1 = os.path.join(module_dir, '../../djangoProject1/poll/stopwords.txt')
    with open(stopword1, encoding='utf-8')as f:
        for line in f.readlines():
            stopwords.append(line.strip())

    doc = re.sub(remove_chars, "", ''.join(doc.split()))
    doc = jieba.lcut(doc)
    if ngram:
        for i in range(len(doc)-1):
            doc.append(doc[i]+doc[i+1])
    return [word for word in doc if word not in stopwords]

def process_data(text, tokenizer, max_len):
    tok_text = tokenizer.encode(text)
    input_ids_orig = tok_text.ids[1:-1]
    text_offsets = tok_text.offsets[1:-1]

    input_ids = [101] + input_ids_orig + [102]
    token_type_ids = [1] + [1] * (len(input_ids_orig) + 1)
    for i in range(len(input_ids)):
        token_type_ids[i]=0
        if input_ids[i]==102:
                       break
    mask = [1] * len(token_type_ids)
    text_offsets = [(0, 0)] + text_offsets + [(0, 0)]

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        text_offsets = text_offsets + ([(0, 0)] * padding_length)
    else:
        input_ids=input_ids[:max_len]
        mask=mask[:max_len]
        token_type_ids=token_type_ids[:max_len]
        text_offsets=text_offsets[:max_len]
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'orig_text': text,
        'offsets': text_offsets
    }


def predict(
        original_text,
        idx_start,
        idx_end,
        offsets
):
    if idx_end < idx_start:
        idx_end = idx_start

    filtered_output = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_text[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += " "

    return filtered_output.strip()

class MyDataset:
    def __init__(self, df):
        self.df = df
        self.text = df.text.values
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        data = process_data(
            self.text[item],
            config.TOKENIZER,
            config.MAX_LEN
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'orig_text': data["orig_text"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }

def test_fn(data_loader, model, device):
    # model = Model(config)
    # model.to(device)
    #     # model=nn.DataParallel(model,device_ids=[0,1])
    # model.load_state_dict(torch.load('/home/zhuhao/project/governmentqa/model/model/model1.pt'))
    model.eval()
    preds = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            orig_text = d["orig_text"]
            offsets = d["offsets"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            #             print(ids)
            outputs_start, outputs_end, output = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids,
            )
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().numpy()
            is_ans = torch.softmax(output, dim=1)[:, 1].cpu().numpy()

            for px, text in enumerate(orig_text):

                output_sentence = predict(
                    original_text=text,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )

                preds.append([output_sentence, is_ans[px]])
    return preds


def predicting(question):
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, '../../djangoProject1/poll/context_fin.csv')
    context = pd.read_csv(file_path)
    # context2 = pd.read_csv('context.csv')
    # context2[['省','市','区']] = cpca.transform(context2.text.values)[['省','市','区']]
    # context[['省','市','区']] = pd.merge(context, context2, on='docid')[['省','市','区']]
    # context.fillna('',inplace=True)
    # context['addr'] = (context.省+context.市+context.区).astype(str)
    # docs = (context['addr'] + context['text']).values

    # texts = [clean(doc) for doc in tqdm(docs)]
    # bm25 = BM25Okapi(texts)
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, '../../djangoProject1/poll/bm25_cache.bin')
    bm25 = pickle.loads(open(file_path, 'rb').read())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device('cuda')
    model = Model(config)
    model.to(device)
    model=nn.DataParallel(model,device_ids=[0,1])
    # model = nn.DataParallel(model, device_ids=[0])
    # model.load_state_dict(torch.load('../temp_data/Trainmodel.pt'))
    module_dir = os.path.dirname(__file__)
    Trainmodel= os.path.join(module_dir, '../../djangoProject1/poll/Trainmodel.pt')
    model.load_state_dict(torch.load(Trainmodel, map_location='cpu'),False)
    # model = torch.load('../temp_data/Trainmodel.pt', map_location='cpu')
    # globals(model,bm25)

    # question = input()
    question_temp = clean(question)
    scores = bm25.get_scores(question_temp)
    values, indices = torch.tensor(scores).topk(50, dim=0, largest=True, sorted=True)

    indices = indices.numpy()
    BM25_ans = pd.DataFrame(columns=['question', 'candId', 'candText'])
    for idx in indices:
        candText = context.loc[idx, 'text']
        candId = context.loc[idx, 'paraid']
        BM25_ans = BM25_ans.append([{'question': question, 'candId': candId, 'candText': candText}], ignore_index=True)

    BM25_ans['text'] = BM25_ans.question + '[SEP]' + BM25_ans.candText
    test_ans = BM25_ans
    test_dataset = MyDataset(BM25_ans)
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False
    )

    preds = test_fn(data_loader, model, device)
    test_ans['pred'] = np.array(preds)[:, 0]
    test_ans['score'] = np.array(preds)[:, 1]

    tmp = test_ans.sort_values(by='score', ascending=False).reset_index()
    j = 0
    while len(tmp.loc[j, 'pred'].replace('[SEP]', '')) == 0:
        j += 1
        if j == 50:
            j = 0
            break

    tmp.loc[j, 'pred'] = tmp.loc[j, 'pred'].replace('[SEP]', '')
    if len(tmp.loc[j, 'pred']):
        final_ans = tmp[['question', 'candId', 'pred', 'score']].loc[j:j]
    else:
        tmp.loc[j, 'pred'] = tmp.loc[j, 'candText']
        final_ans = tmp[['question', 'candId', 'pred', 'score']].loc[j:j]

    print(final_ans['pred'])
    print(final_ans['candId'])

    return final_ans.loc[0,'pred'],final_ans.loc[0,'candId']

def re_text(candId):
    docs = []
    module_dir = os.path.dirname(__file__)
    csv = os.path.join(module_dir, '../../djangoProject1/poll/NCPPolicies_context_20200301.csv')
    with open(csv, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t', 1)
            docs.append(line)
    docs = pd.DataFrame(docs[1:])
    docs.columns = ['docid', 'text']
    text=docs[docs['docid']==candId[:-3]]['text']
    return candId[:-3],text


