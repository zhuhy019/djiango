# -*- coding = utf-8 -*-
# @Time ： 2021/11/7 13:56
# @Author : 朱豪
# @File : train.py
# @Software : PyCharm

import pandas as pd
import re
from rank_bm25 import BM25Okapi
import torch
import numpy as np
import jieba_fast
from tqdm import tqdm
import cpca

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import BertModel, BertTokenizer
import tokenizers
import rouge



#BM25 召回
context = pd.read_csv('/context_fin.csv')
context2 = pd.read_csv('/context.csv')
train = pd.read_csv('../temp_data/train_fin.csv')

stopwords = []
with open('/stopwords.txt', encoding='utf-8')as f:
    for line in f.readlines():
        stopwords.append(line.strip())

remove_chars = '[·’!"\#$%&\'()＃！（）*+,-.。/:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
def clean(doc, ngram=True):
    doc = re.sub(remove_chars, "", ''.join(doc.split()))
    doc = jieba_fast.lcut(doc)
    if ngram:
        for i in range(len(doc) - 1):
            doc.append(doc[i] + doc[i + 1])
    return [word for word in doc if word not in stopwords]

context2[['省','市','区']] = cpca.transform(context2.text.values)[['省','市','区']]
context[['省','市','区']] = pd.merge(context, context2, on='docid')[['省','市','区']]
context.fillna('',inplace=True)
context['addr'] = (context.省+context.市+context.区).astype(str)
docs = (context['addr'] + context['text']).values
texts = [clean(doc) for doc in tqdm(docs)]
bm25 = BM25Okapi(texts)
#储存BM25模型
import pickle
f = open("/bm25_cache.bin", 'wb')
pickle.dump(bm25, f)
f.close()
train = train.dropna(subset=['paraid']).reset_index()
train['text'] = pd.merge(train,context,on= 'paraid')['text']
cnt = 0

train['candicate'] = ''
train2 = pd.DataFrame(columns=['question', 'answer', 'paraid', 'candId', 'candText', 'isAns'])
cnt = 0

tk = tqdm(range(len(train)))
for i in tk:

    question = train.loc[i, 'question']
    docid = train.loc[i, 'docid']
    paraid = train.loc[i, 'paraid']
    answer = train.loc[i, 'answer']
    text = train.loc[i, 'answer']

    question_tmp = clean(question)
    scores = bm25.get_scores(question_tmp)

    values, indices = torch.tensor(scores).topk(50, dim=0, largest=True, sorted=True)
    indices = indices.numpy()
    if paraid in context.loc[indices, 'paraid'].values:
        cnt += 1
        for idx in indices:
            candId = context.loc[idx, 'paraid']
            candText = context.loc[idx, 'text']
            if paraid == context.loc[idx, 'paraid']:
                train2 = train2.append([{'question': question, 'answer': answer, 'paraid': paraid, 'candId': candId,
                                         'candText': candText, 'isAns': 1}], ignore_index=True)
            else:
                train2 = train2.append([{'question': question, 'answer': answer, 'paraid': paraid, 'candId': candId,
                                         'candText': candText, 'isAns': 0}], ignore_index=True)
    else:
        train2 = train2.append([{'question': question, 'answer': answer, 'paraid': paraid, 'candId': paraid,
                                 'candText': text, 'isAns': 1}], ignore_index=True)
        for idx in indices[:-1]:
            candId = context.loc[idx, 'paraid']
            candText = context.loc[idx, 'text']
            train2 = train2.append([{'question': question, 'answer': answer, 'paraid': paraid, 'candId': candId,
                                     'candText': candText, 'isAns': 0}], ignore_index=True)

    tk.set_postfix(cnt=cnt)


#['question','answer',''paraid','candId','candText','isAns']
train2.to_csv('train_fin3.csv',index=False)



#Bert

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class config:
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8
    EPOCHS = 10
#     BERT_PATH = "chinese_wwm_ext_pytorch/"
    BERT_PATH = "chinese-macbert-large/"

    TOKENIZER = tokenizers.BertWordPieceTokenizer(
        f"{BERT_PATH}/vocab.txt",
        lowercase=True
    )

device =torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
rouge = rouge.Rouge()
train_df = pd.read_csv('../temp_data/train_fin3.csv').dropna()

train_df['text'] = train_df.question+'[SEP]'+train_df.candText
train_df['selected_text'] = train_df.answer
train_df['label'] = train_df.isAns
train_df['id'] = 0
for i in range(0,len(train_df)//50):
    train_df.loc[i*50:i*50+50,'id'] = i


def process_data(text, selected_text, tokenizer, max_len):
    tokenizer.enable_truncation(max_length=max_len)

    len_st = len(selected_text)
    idx0 = None
    idx1 = None
    #     答案在text中的位置区间
    for ind in (i for i, e in enumerate(text) if e == selected_text[0]):
        if text[ind: ind + len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(text)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1

    tok_text = tokenizer.encode(text)
    input_ids_orig = tok_text.ids[1:-1]
    text_offsets = tok_text.offsets[1:-1]

    target_idx = []
    for j, (offset1, offset2) in enumerate(text_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)

    if len(target_idx):
        targets_start = target_idx[0]
        targets_end = target_idx[-1]
    else:
        targets_start = -1
        targets_end = -1

    input_ids = [101] + input_ids_orig + [102]
    token_type_ids = [1] + [1] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    text_offsets = [(0, 0)] + text_offsets + [(0, 0)]
    targets_start += 1
    targets_end += 1

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        text_offsets = text_offsets + ([(0, 0)] * padding_length)

    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_text': text,
        'orig_selected': selected_text,
        'offsets': text_offsets
    }


class MyDataset:
    def __init__(self, df):
        self.df = df
        self.text = df.text.values
        self.selected_text = df.selected_text.values
        self.label = df.label.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        data = process_data(
            self.text[item],
            self.selected_text[item],
            config.TOKENIZER,
            config.MAX_LEN
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_text': data["orig_text"],
            'orig_selected': data["orig_selected"],
            'label': self.label[item],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_PATH,return_dict=False)
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

def loss_fn(start_logits, end_logits, start_positions, end_positions, output, label):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    labeled_loss = loss_fct(output, label)
    total_loss = (start_loss + end_loss + labeled_loss*0.5)
    return total_loss, labeled_loss

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def predict(
        original_text,
        target_string,
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


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    #     fgm = FGM(model)
    losses = AverageMeter()
    label_losses = AverageMeter()
    accs = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for step, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        label = d["label"]
        orig_selected = d["orig_selected"]
        orig_text = d["orig_text"]
        offsets = d["offsets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        label = label.to(device, dtype=torch.long)

        model.zero_grad()
        outputs_start, outputs_end, output = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )
        with autocast():
            loss, label_loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end, output, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

        acc = (output.argmax(1) == label).sum() / ids.size(0)

        losses.update(loss.item(), ids.size(0))
        label_losses.update(label_loss.item(), ids.size(0))
        accs.update(acc.item(), ids.size(0))
        tk0.set_postfix(all_loss=losses.avg, cls_loss=label_losses.avg, cls_acc=accs.avg)


def eval_fn(valid, data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    label_losses = AverageMeter()
    #     accs = AverageMeter()

    preds = []

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            label = d["label"]
            orig_selected = d["orig_selected"]
            orig_text = d["orig_text"]
            offsets = d["offsets"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            label = label.to(device, dtype=torch.long)

            outputs_start, outputs_end, output = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids,
            )
            loss, label_loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end, output, label)

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().numpy()
            is_ans = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            #             acc = (output.argmax(1)==label).sum()/ids.size(0)
            for px, text in enumerate(orig_text):
                selected_text = orig_selected[px]

                output_sentence = predict(
                    original_text=text,
                    target_string=selected_text,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )

                preds.append([output_sentence, is_ans[px]])

            losses.update(loss.item(), ids.size(0))
            label_losses.update(label_loss.item(), ids.size(0))

            #             accs.update(acc.item(), ids.size(0))
            tk0.set_postfix(all_loss=losses.avg, cls_loss=label_losses.avg)

    valid['pred'] = np.array(preds)[:, 0]
    valid['score'] = np.array(preds)[:, 1]

    final = []
    for i in range(3997, 4997):
        j = 0
        tmp = valid[valid.id == i].sort_values(by='score', ascending=False).reset_index()

        while len(tmp.loc[j, 'pred'].replace('[SEP]', '')) == 0:
            j += 1
            if j == 50:
                j = 0
                break

        tmp.loc[j, 'pred'] = tmp.loc[j, 'pred'].replace('[SEP]', '')
        if len(tmp.loc[j, 'pred']):
            final.append(tmp[['question', 'answer', 'pred', 'score']].loc[j:j])
        else:
            tmp.loc[j, 'pred'] = tmp.loc[j, 'candText']
            final.append(tmp[['question', 'answer', 'pred', 'score']].loc[j:j])

    final = pd.concat(final, 0)
    scores = 0
    for q, a, pred, score in final.values:
        source, target = ' '.join(pred), ' '.join(a)
        scores += rouge.get_scores(hyps=source, refs=target)[0]['rouge-l']['f']
    return losses.avg, scores / 1000


model = Model(config)
model.to(device)
model=nn.DataParallel(model,device_ids=[0,1])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
scaler = GradScaler()

best_score = 0

for epoch in range(config.EPOCHS):
    train = train_df.loc[:3997 * 50 - 1].reset_index(drop=True)
    sampled_data = []
    for i in range(3997):
        tmp = train[train.id == i]
        positive_data = tmp[tmp.label == 1]
        negative_data = tmp[tmp.label == 0].sample(1)
        sampled_data.append(pd.concat([positive_data, negative_data], 0))
    train = pd.concat(sampled_data, 0)
    valid = train_df.loc[3997*50:].reset_index(drop=True)

    train_dataset = MyDataset(train)
    valid_dataset = MyDataset(valid)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            shuffle=False
    )

    train_fn(train_dataloader, model, optimizer, device, scheduler=None)
    val_loss, val_score = eval_fn(valid, valid_dataloader, model, device)
#     print(val_score)
    if val_score > best_score:
        best_score = val_score
    torch.save(model.state_dict(), 'model1.pt')


# final.to_csv('my_pred1.csv', index=False)
# valid.to_csv('all_my_pred1.csv', index=False)

