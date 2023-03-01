#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/11/14 10:53
# software: PyCharm


import json
import numpy as np
import torch
from keras_preprocessing.sequence import pad_sequences

from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import trange, tqdm
from torch.optim import AdamW
from transformers import BertConfig, BertForSequenceClassification
from my_bert import BertAdapterForSequenceClassification, SkipBertAdapterForSequenceClassification
from transformers import BertTokenizer
from transformers.adapters import AdapterConfig
from transformers import logging

import method
from skipbert.modeling import SkipBertForSequenceClassification, SkipBertModel, ShallowSkipping

data_path = './data/text_classification/chemprot/'  # Task Path

label_num = method.label2num(data_path)  
num_token = len(label_num)  


def data_read(bert_model, data_type):
    """
    :param data_type: train, test,dev
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


def data_load(bert_model, file_type, device, batch_size):
    """
    :param file_type: 
    :param device:
    :param batch_size:
    :return: DataLoader
    """
    inputs, labels, masks = data_read(bert_model, file_type) 

    inputs = torch.tensor(inputs).to(device)
    labels = torch.tensor(labels).to(device)
    masks = torch.tensor(masks).to(device)

    data = TensorDataset(inputs, masks, labels)
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


def train_classier(bert_model, epochs, device, batch_size, learning_rate, seed,
                   use_adapter, isLinear=False, skip_flag=False, feature_flag=False):
    """
    :param bert_model:
    :param epochs:
    :param device:
    :param batch_size:
    :param learning_rate:
    :param seed:
    :param use_adapter: 
    :param isLinear: 
    :param skip_flag: 
    :param feature_flag: 
    """

    method.setup_seed(seed)

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
            if feature_flag:
                model = method.feature_base(model)

    if skip_flag:
        model.freeze_shallow_layers()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    train_dataloader = data_load(bert_model, 'train', device, batch_size)
    validation_dataloader = data_load(bert_model, 'dev', device, batch_size)
    test_dataloader = data_load(bert_model, 'test', device, batch_size)
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
        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
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


def mix_model_test(shallow_model, new_model, test_dataloader, device, model_type,
                   new_corpus_id):
    new_model.eval()
    old_skiping, old_skipmodel, new_skiping, new_skipmodel = shallow_model

    n_true = 0
    n_total = 0
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, labels = batch

        trigrams_id = method.input_ids_to_tri_grams(input_ids.cpu().numpy())
        trigrams_id_string = method.id2string(trigrams_id)

        with torch.no_grad():
            token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long, device=device)

            old_res = old_skiping(input_ids, token_type_ids=token_type_ids,
                                  attention_mask=input_mask, model=old_skipmodel)
            new_res = new_skiping(input_ids, token_type_ids=token_type_ids,
                                  attention_mask=input_mask, model=new_skipmodel)

            old_tri_states = old_res[1]
            new_tri_states = new_res[1]

            mix_tri_states = method.merge_trigram_states_by_input_id(old_tri_states, new_tri_states,
                                                                     trigrams_id_string, new_corpus_id)

            logits = new_model(input_ids, token_type_ids=None, attention_mask=input_mask,
                               trigram_states=mix_tri_states)[0]

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        n_true += np.sum(pred_flat == labels_flat)
        n_total += len(labels_flat)

    accuracy = n_true / n_total
    print(model_type + "Accuracy: {}".format(accuracy))

    return accuracy


def mix_fine_tune(old_model_path, new_model_path, learning_rate, epochs,
                  device, batch_size, seed, new_corpus_id, use_adapter):

    method.setup_seed(seed)
    old_skipmodel = SkipBertModel.from_pretrained(old_model_path).eval()
    method.setup_seed(seed)
    new_skipmodel = SkipBertModel.from_pretrained(new_model_path).eval()
    old_skipmodel.to(device)
    new_skipmodel.to(device)

    old_skiping = ShallowSkipping(old_skipmodel).eval()
    new_skiping = ShallowSkipping(new_skipmodel).eval()

    shallow_model = old_skiping, old_skipmodel, new_skiping, new_skipmodel

    if use_adapter:
        method.setup_seed(seed)

        new_modelConfig = BertConfig.from_pretrained(new_model_path)
        new_modelConfig.has_adapter = use_adapter 
        new_modelConfig.isLinear = False
        new_modelConfig.num_labels = num_token 

        new_model = SkipBertAdapterForSequenceClassification.from_pretrained(new_model_path, config=new_modelConfig)

        new_model.freeze_model(True)  # freeze all params
        # unfreeze adapter params
        adapter_param = ["adapter_fi", "adapter_se"]
        adapter_param_list = [p for n, p in new_model.named_parameters() if any(nd in n for nd in adapter_param)]
        for param in adapter_param_list:
            param.requires_grad = True
    else:

        method.setup_seed(seed)

        new_modelConfig = BertConfig.from_pretrained(new_model_path)
        new_modelConfig.num_labels = num_token  
        new_model = SkipBertForSequenceClassification.from_pretrained(new_model_path, config=new_modelConfig)

    new_model.freeze_shallow_layers()
    new_model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, new_model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    train_dataloader = data_load(new_model_path, 'train', device, batch_size)
    validation_dataloader = data_load(new_model_path, 'dev', device, batch_size)
    test_dataloader = data_load(new_model_path, 'test', device, batch_size)

    val_acc = [] 
    test_acc = [] 

    for name, param in new_model.named_parameters():
        if param.requires_grad:
            print(name, end=" ")
            print(param.requires_grad)

    for i in range(epochs):
        print("Epochs:%d/%d" % ((i + 1), epochs))
        new_model.train()

        tr_loss = 0
        nb_tr_steps = 0
        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, labels = batch
            token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long, device=device)

            trigrams_id = method.input_ids_to_tri_grams(input_ids.cpu().numpy())
            trigrams_id_string = method.id2string(trigrams_id)

            old_res = old_skiping(input_ids, token_type_ids=token_type_ids,
                                  attention_mask=input_mask, model=old_skipmodel)
            new_res = new_skiping(input_ids, token_type_ids=token_type_ids,
                                  attention_mask=input_mask, model=new_skipmodel)

            old_tri_states = old_res[1]
            new_tri_states = new_res[1]

            mix_tri_states = method.merge_trigram_states_by_input_id(old_tri_states, new_tri_states,
                                                                     trigrams_id_string, new_corpus_id)

            mix_res = new_model(input_ids, token_type_ids=None, attention_mask=input_mask,
                                labels=labels, trigram_states=mix_tri_states)

            loss = mix_res[0]

            # print(mix_res[1].equal(new_res[1]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_steps += 1

        scheduler.step()
        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        v_acc = mix_model_test(shallow_model, new_model, validation_dataloader, device, 'Val', new_corpus_id)
        t_acc = mix_model_test(shallow_model, new_model, test_dataloader, device, 'Test', new_corpus_id)
        val_acc.append(v_acc)
        test_acc.append(t_acc)

    return val_acc, test_acc


def main():
    epochs = 15
    batch_size = 32
    learning_rate = 7e-7
    seed = 40
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trigram_path_5 = './data/trigram/Medicine/trigram_0.5.txt'
    new_corpus_id_5 = []
    with open(trigram_path_5, 'r') as record_file:
        for x in tqdm(record_file.readlines()):
            y = x.strip()
            new_corpus_id_5.append(y)
    new_corpus_id_5 = np.array(new_corpus_id_5)

    bert_model0 = './model/skip-mlm-new/'  
    print("SkipBert Training------")
    val_acc0, test_acc0 = train_classier(bert_model0, epochs, device, batch_size, learning_rate, seed,
                                         use_adapter=False, skip_flag=True)
    bert_model1 = './model/Medicine/
    print("Biology Training------")
    val_acc1, test_acc1 = train_classier(bert_model1, epochs, device, batch_size, learning_rate, seed,
                                         use_adapter=False, skip_flag=True)
    print("Fre 50% Training")
    val_acc2, test_acc2 = mix_fine_tune(bert_model0, bert_model1, learning_rate, epochs,
                                        device, batch_size, seed, new_corpus_id_5, use_adapter=False)

    # print("Mix 1 Training")
    # val_acc3, test_acc3 = mix_fine_tune(bert_model0, bert_model1, learning_rate,
    #                                     epochs, device, batch_size, seed, -1)

    # print("Mix 0.9 Training")
    # val_acc4, test_acc4 = mix_fine_tune(bert_model0, bert_model1, learning_rate,
    #                                     epochs, device, batch_size, seed, 0.0176)

    legend = ['SkipBert', 'Medicine', 'Fre 50%']
    Val_Acc = [val_acc0, val_acc1, val_acc2]
    Test_Acc = [test_acc0, test_acc1, test_acc2]

    # method.plot_res('rct-20k Val Acc', legend, Val_Acc)
    # method.plot_res('rct-20k Test Acc', legend, Test_Acc)
    method.get_test_acc(legend, Val_Acc, Test_Acc)
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i], end=" ")
        print("Max Test Acc: ")
        print(max(Test_Acc[i]))


if __name__ == "__main__":
    main()
