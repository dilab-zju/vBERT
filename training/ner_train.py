#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/10/8 10:10
# software: PyCharm
import json
import os
import random

import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, random_split
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig, BertTokenizer
import torch
import method

from my_bert import BertAdapterForTokenClassification, SkipBertAdapterForTokenClassification
from skipbert.modeling import SkipBertForTokenClassification

data_path = './data/SubTask/ner/SciERC/'  # task data path
unique_tags, tag2id, id2tag = method.ner_label(data_path)


class NerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def data_read(data_type):
    sentencses = []
    labels = []
    path = data_path + data_type + '.txt'
    print("%s data loading------" % data_type)

    with open(path, 'r', encoding='utf-8') as file:
        tmp_words = []
        tmp_label = []
        for line in tqdm(file.readlines()):
            if 'DOCSTART' in line:
                continue
            else:
                if len(line) == 1:
                    sentencses.append(tmp_words)
                    labels.append(tmp_label)
                    tmp_words = []
                    tmp_label = []
                else:
                    tmp = line.split()
                    tmp_words.append(tmp[0])
                    tmp_label.append(tmp[-1])

    return sentencses, labels


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    # print(labels)
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        if len(doc_labels) >= 510:  
            doc_labels = doc_labels[:510]
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def data_load(model_path, batch_size, data_type, split_zie=0):
    # model_path = './model/bert-base-uncased/'
    texts, tags = data_read(data_type)

    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                          truncation=True, max_length=512)

    labels = encode_tags(tags, encodings)
    encodings.pop("offset_mapping") 

    dataset = NerDataset(encodings, labels)

    if split_zie != 0:
       
        if len(dataset) > split_zie:
            data_size = split_zie
        else:
            data_size = len(dataset)

        print("Choose " + str(data_size) + " Data for " + data_type)
        dataset, remain_data = random_split(dataset=dataset, lengths=[data_size, len(dataset) - data_size])

    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader


def token2span(token_list):

    span_list = []
    span_tuple = []
    merge_flag = False
    for i in range(len(token_list)):
        if token_list[i][0] == 'B':
            if merge_flag: 
                span_tuple.append(i - 1) 
                span_tuple.append(token_list[i - 1][2:]) 
                span_list.append(tuple(span_tuple))  
                span_tuple = []

            merge_flag = True
            span_tuple.append(i)  

        elif token_list[i][0] == 'I': 
            if merge_flag: 
                continue
            else:  
                span_tuple.append(i) 
                span_tuple.append(i)  
                span_tuple.append(token_list[i][2:]) 
                span_list.append(tuple(span_tuple))  
                span_tuple = []

        else: 
            if merge_flag:  
                merge_flag = False
                span_tuple.append(i - 1)
                span_tuple.append(token_list[i - 1][2:]) 
                span_list.append(tuple(span_tuple)) 
                span_tuple = []

    return span_list


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def model_test(model, test_dataloader, device, model_type):

    model.eval()
    pred_token_list = []
    label_token_list = []

    for batch in test_dataloader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
        b_preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
        b_label_ids = b_labels.cpu().numpy().tolist()
        for i in range(len(b_label_ids)):
            pred = b_preds[i][1:-1]
            label = b_label_ids[i][1:-1]
            pred_list = [id2tag[x] for (x, y) in zip(pred, label) if y != -100]
            label_list = [id2tag[x] for x in label if x != -100]
            # print(pred_list)
            pred_token_list.extend(pred_list)
            label_token_list.extend(label_list)
    # print("-------------------")

    pred_span_set = set(token2span(pred_token_list))
    label_span_set = set(token2span(label_token_list))
    # print(pred_span_set)
    t_p = len(pred_span_set & label_span_set)

    print("t_p:" + str(t_p))
    if t_p == 0:
        F1 = 0
    else:
        precision = t_p / len(pred_span_set)
        recall = t_p / len(label_span_set)
        F1 = (2 * precision * recall) / (precision + recall)

    print(model_type + "F1: {}".format(F1))

    return F1


def train(model_path, epochs, device, batch_size, learning_rate, use_adapter,
          seed=40, isLinear=False, skip_flag=False, feature_flag=False):

    setup_seed(seed)

    train_dataloader = data_load(model_path, batch_size, 'train')
    validation_dataloader = data_load(model_path, batch_size, 'dev')
    test_dataloader = data_load(model_path, batch_size, 'test')

    if use_adapter:
  
        modelConfig = BertConfig.from_pretrained(model_path)
        modelConfig.has_adapter = use_adapter  
        modelConfig.isLinear = isLinear
        modelConfig.num_labels = len(unique_tags)  
        if skip_flag:
            model = SkipBertAdapterForTokenClassification.from_pretrained(model_path, config=modelConfig)
        else:
            model = BertAdapterForTokenClassification.from_pretrained(model_path, config=modelConfig)

        model.freeze_model(True)  # freeze all params
        # unfreeze adapter params
        adapter_param = ["adapter_fi", "adapter_se"]
        adapter_param_list = [p for n, p in model.named_parameters() if any(nd in n for nd in adapter_param)]
        for param in adapter_param_list:
            param.requires_grad = True
    else:
        modelConfig = BertConfig.from_pretrained(model_path)
        modelConfig.num_labels = len(unique_tags) 
        if skip_flag:
            model = SkipBertForTokenClassification.from_pretrained(model_path, config=modelConfig)
        else:
            model = BertForTokenClassification.from_pretrained(model_path, config=modelConfig)

    if skip_flag:
        model.freeze_shallow_layers()

    if feature_flag:
        model = method.feature_base(model)  

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    model.to(device)
    val_acc = []
    test_acc = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    # model_test(model, validation_dataloader, device, 'Val ')
    for i in range(epochs):
        print("Epochs:%d/%d" % ((i + 1), epochs))
        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for batch in train_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            optimizer.zero_grad()
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

            nb_tr_steps += 1

        scheduler.step()
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        acc = model_test(model, validation_dataloader, device, 'Val ')
        val_acc.append(acc)
        acc1 = model_test(model, test_dataloader, device, 'Test ')
        test_acc.append(acc1)

    return val_acc, test_acc


def find_best_param(l_r):
    model_path = './model/Computer/'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    epoch = 6
    batch_list = []
    test_acc_list = []
    val_acc_list = []
    Max_test_acc = []

    for i in range(epoch):
        batch_size = 2 ** (epoch - i)  
        print("Computer %d training:" % batch_size)
        val_acc, test_acc = train(model_path, 15, device, batch_size, learning_rate=l_r,
                                  use_adapter=False, skip_flag=True, feature_flag=True)
        batch_list.append(batch_size)
        index = val_acc.index(max(val_acc)) 
        Max_test_acc.append(max(test_acc))

        val_acc_list.append(val_acc[index])
        test_acc_list.append(test_acc[index])

    for i in range(len(batch_list)):
        print("Batch Size: %d" % batch_list[i], end=" ")
        print("Val:")
        print(val_acc_list[i])
        print("Test: ")
        print(test_acc_list[i])
    best_batch_size = batch_list[test_acc_list.index(max(test_acc_list))]
    cor_test_acc = max(test_acc_list)
    print("---------------------")
    print(best_batch_size)
    print(cor_test_acc)
    print("Max_test_acc---------------------")
    print(batch_list[Max_test_acc.index(max(Max_test_acc))])
    print(max(Max_test_acc))
    print("------------------------------------------------------------------------------------------------------")

    return best_batch_size, cor_test_acc


def find_lr():
    batch_size_list = []
    test_acc_test = []
    lr_test = []
    for i in range(1, 10):
        l_r = i / 1000
        lr_test.append(l_r)
        batch_size, test_acc = find_best_param(l_r)
        batch_size_list.append(batch_size)
        test_acc_test.append(test_acc)

    print("---------------------")
    for i in range(len(batch_size_list)):
        print(lr_test[i])
        print(batch_size_list[i])
        print(test_acc_test[i])
    print("--------------------------")
    print(max(test_acc_test))
    print(batch_size_list[test_acc_test.index(max(test_acc_test))])
    print(lr_test[test_acc_test.index(max(test_acc_test))])


def average_accuracy():
    task = data_path.split('/')[-2] 
    epochs = 15
    batch_size = 16
    learning_rate = 5e-5
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    accuracy_list = []
    avg_num = 1000

    model_name = 'bert-base-uncased'
    # model_name = 'DistillBert'
    # model_name = 'TinyBert_4'
    # model_name = 'TinyBert_6'
    # model_name = 'Computer'

    bert_model1 = './model/' + model_name + '/' 

    use_adapter = True
    skip_flag = False
    feature_flag = False

    save_name = str(avg_num) + '-' + model_name
    if use_adapter:
        save_name = save_name + "-adapter"
        learning_rate = 10 * learning_rate
    if feature_flag:
        save_name = save_name + "-feature"
        learning_rate = 10 * learning_rate

    print("-------------------------")
    print(task)
    print(batch_size)
    print(learning_rate)
    print(device)
    t0 = method.time_beijing()
    print("--------------------------")

    for i in range(avg_num):
        print(save_name + " &d epoch training------" % (i + 1))
        val_acc, test_acc = train(bert_model1, epochs, device, batch_size, learning_rate,
                                  seed=i, use_adapter=use_adapter, skip_flag=skip_flag, feature_flag=feature_flag)

        index = val_acc.index(max(val_acc))
        print('Highest Val Acc:%f,Epoch:%d' % (val_acc[index], index), end=', \t')
        print('Corresponding Test Acc:%f' % test_acc[index])
        accuracy_list.append(test_acc[index])

    print("-------------------------")
    for i in range(avg_num):
        print(accuracy_list[i])

    print("-------------------------")
    print("Avg Accuracy：")
    print(sum(accuracy_list) / len(accuracy_list))
    print("-------------------------")
    t1 = method.time_beijing()
    print("Progrem Start:" + str(t0))
    print("Progrem End:" + str(t1))

    print("Save Result!")
    save_path = './outputs/AvgAcc/' + task + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(accuracy_list, save_path + save_name + '.pt')


def main():
    epochs = 15
    batch_size = 2
    learning_rate = 7e-4
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("-------------------------")
    print(batch_size)
    print(learning_rate)
    print(device)
    print("--------------------------")

    bert_model0 = './model/bert-base-uncased/' 
    print("Bert Training------")
    val_acc0, test_acc0 = train(bert_model0, epochs, device, 32, 4e-5,
                                use_adapter=False, skip_flag=False)

    bert_model1 = './model/Computer/'  # you need get this model by further pretraining
    print("Computer SkipBert Adapter Training------")
    val_acc1, test_acc1 = train(bert_model1, epochs, device, batch_size, learning_rate,
                                use_adapter=True, skip_flag=True)
    
    legend = ['Bert', 'Computer Adapter'']
    Val_Acc = [val_acc0, val_acc1]
    Test_Acc = [test_acc0, test_acc1]

    method.plot_res('Val Acc', legend, Val_Acc)
    method.plot_res('Test Acc', legend, Test_Acc)
    method.get_test_acc(legend, Val_Acc, Test_Acc)
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i], end=" ")
        print("Max Test Acc: ")
        print(max(Test_Acc[i]))


if __name__ == "__main__":
    # main()
    # find_best_param()
    # find_lr()
    average_accuracy()

