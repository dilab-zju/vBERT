#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/5/6 14:53
# software: PyCharm

import json
import os

import numpy as np
import torch
from keras_preprocessing.sequence import pad_sequences

from torch.utils.data import TensorDataset, RandomSampler, DataLoader, random_split
from tqdm import trange, tqdm
from torch.optim import AdamW
from transformers import BertConfig, BertForSequenceClassification

from my_bert import BertAdapterForSequenceClassification, SkipBertAdapterForSequenceClassification
from transformers import BertTokenizer
from transformers.adapters import AdapterConfig
from transformers import logging

import method
from skipbert.modeling import SkipBertForSequenceClassification

logging.set_verbosity_error()

data_path = './data/SubTask/cls/sci-cite/'  # downstream task path

label_num = method.label2num(data_path) 
num_token = len(label_num) 


def data_read(bert_model, data_type):
    """
    :param data_type: train, test, dev
    :return: input_ids, labels, attention_masks
    """
    sentencses = []
    labels = []
    path = data_path + data_type + '.txt'
    print("%s data loading------" % data_type)
    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines()):
            dic = json.loads(line)
            sent = '[CLS] ' + dic['text'] + ' [SEP]' 
            label_token = dic['label']  
            label = int(label_num[label_token])  

            sentencses.append(sent)
            labels.append(label)

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]

    MAX_LEN = 128

    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    return input_ids, labels, attention_masks


def data_load(bert_model, file_type, device, batch_size, split_zie=0):
    """
    :param bert_model: 
    :param file_type: 
    :param device: 
    :param batch_size: 
    :param split_zie:
    :return: DataLoader
    """
    inputs, labels, masks = data_read(bert_model, file_type)  

    inputs = torch.tensor(inputs).to(device)
    labels = torch.tensor(labels).to(device)
    masks = torch.tensor(masks).to(device)

    data = TensorDataset(inputs, masks, labels)

    if split_zie != 0:
        if len(data) > split_zie:
            data_size = split_zie
        else:
            data_size = len(data)

        print("Choose " + str(data_size) + " Data for " + file_type)
        data, remain_data = random_split(dataset=data, lengths=[data_size, len(data) - data_size])

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


def model_test(model, test_dataloader, device, model_type):
    """
    :param model: 
    :param test_dataloader: 
    :param device: 
    :param model_type:
    """
    model.eval()
    n_true = 0
    n_total = 0
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        n_true += np.sum(pred_flat == labels_flat)
        n_total += len(labels_flat)

    accuracy = n_true / n_total
    print(model_type + "Accuracy: {}".format(accuracy))

    return accuracy


def train_classier(bert_model, epochs, device, batch_size, learning_rate, use_adapter,
                   seed=40, isLinear=False, skip_flag=False, feature_flag=False):
    """
    :param bert_model:
    :param epochs: 
    :param device:
    :param batch_size:
    :param learning_rate: 
    :param use_adapter:
    :param seed:
    :param isLinear: 
    :param skip_flag:
    :param feature_flag: 
    """

    method.setup_seed(seed)

    train_dataloader = data_load(bert_model, 'train', device, batch_size)
    validation_dataloader = data_load(bert_model, 'dev', device, batch_size)
    test_dataloader = data_load(bert_model, 'test', device, batch_size)

    if use_adapter: 
        modelConfig = BertConfig.from_pretrained(bert_model)
        modelConfig.has_adapter = use_adapter  
        modelConfig.isLinear = isLinear
        modelConfig.num_labels = num_token 
        if skip_flag:
            model = SkipBertAdapterForSequenceClassification.from_pretrained(bert_model, config=modelConfig)
        else:
            model = BertAdapterForSequenceClassification.from_pretrained(bert_model, config=modelConfig)

        model.freeze_model(True)  # freeze all params
        # unfreeze adapter params
        adapter_param = ["adapter_fi", "adapter_se"]
        adapter_param_list = [p for n, p in model.named_parameters() if any(nd in n for nd in adapter_param)]
        for param in adapter_param_list:
            param.requires_grad = True
    else:
        modelConfig = BertConfig.from_pretrained(bert_model)
        modelConfig.num_labels = num_token 
        if skip_flag:
            model = SkipBertForSequenceClassification.from_pretrained(bert_model, config=modelConfig)
        else:
            model = BertForSequenceClassification.from_pretrained(bert_model, config=modelConfig)

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
            print(name, end=" ")
            print(param.requires_grad)

    for i in range(epochs):
        print("Epochs:%d/%d" % ((i + 1), epochs))
        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()

            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                         labels=b_labels)[0]
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
    model_path = './model/Biology/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch = 6
    batch_list = []
    test_acc_list = []
    val_acc_list = []
    Max_test_acc = []
    print(str(l_r) + "is training-----------")

    for i in range(epoch):
        batch_size = 2 ** (epoch - i) 
        print("Biology size %d training:" % batch_size)
        val_acc, test_acc = train_classier(model_path, 15, device, batch_size, learning_rate=l_r,
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
        l_r = i / 1000000
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    accuracy_list = []
    avg_num = 1000

    # model_name = 'bert-base-uncased'
    # model_name = 'DistillBert'
    # model_name = 'TinyBert_4'
    # model_name = 'TinyBert_6'
    model_name = 'Biology'

    bert_model1 = './model/' + model_name + '/'
    use_adapter = True
    skip_flag = True
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
        print(save_name + " %d Epoch Training------" % (i + 1))
        val_acc, test_acc = train_classier(bert_model1, epochs, device, batch_size, learning_rate,
                                           seed=i, use_adapter=use_adapter, skip_flag=skip_flag,
                                           feature_flag=feature_flag)

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
    print("Progrem Start" + str(t0))
    print("Progrem End：" + str(t1))

    print("Save Result!")
    save_path = './outputs/AvgAcc/' + task + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(accuracy_list, save_path + save_name + '.pt')


def split_test():
    bert_model = './model/bert-base-uncased/' 
    batch_size = 32
    learning_rate = 5e-5
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    loader_list = []
    for i in range(10):
        method.setup_seed(40)

        train_dataloader = data_load(bert_model, 'train', device, batch_size)
        loader_list.append(train_dataloader)

    for i in range(1, 10):
        print(loader_list[i])
        print("--------------------------------------")
        print(loader_list[i-1])
        break


def main():
    epochs = 15
    batch_size = 32
    learning_rate = 5e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("-------------------------")
    print(batch_size)
    print(learning_rate)
    print(device)
    print("--------------------------")

    bert_model0 = './model/bert-base-uncased/'
    print("Bert------")
    val_acc0, test_acc0 = train_classier(bert_model0, epochs, device, 32, 5e-5,
                                         use_adapter=False, skip_flag=False)

    bert_model1 = './model/Biology/'  # you need to get this model by further pretraining
    print("Biology Adapter------")
    val_acc1, test_acc1 = train_classier(bert_model1, epochs, device, batch_size, learning_rate,
                                         use_adapter=True, skip_flag=True)


    legend = ['Bert', 'Biology Adapter']
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
    average_accuracy()


