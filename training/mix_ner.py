#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/11/28 16:52
# software: PyCharm
import json
import random

import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig, BertTokenizer
import torch
import method

from my_bert import BertAdapterForTokenClassification, SkipBertAdapterForTokenClassification
from skipbert.modeling import SkipBertForTokenClassification, SkipBertModel, ShallowSkipping

data_path = './data/ner/SciERC/'  # Task Path
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


def data_load(model_path, batch_size, data_type):
    # model_path = './model/bert-base-uncased/'
    texts, tags = data_read(data_type)

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                          truncation=True, max_length=512)

    labels = encode_tags(tags, encodings)
    encodings.pop("offset_mapping")

    dataset = NerDataset(encodings, labels)
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
          isLinear=False, skip_flag=False, feature_flag=False):
    setup_seed(40)

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
            if feature_flag:
                model = method.feature_base(model) 

    if skip_flag:
        model.freeze_shallow_layers()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    train_dataloader = data_load(model_path, batch_size, 'train')
    validation_dataloader = data_load(model_path, batch_size, 'dev')
    test_dataloader = data_load(model_path, batch_size, 'test')
    model.to(device)
    val_acc = []  
    test_acc = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    model_test(model, validation_dataloader, device, 'Val ')
    for i in range(epochs):
        print("Epochs:%d/%d" % ((i + 1), epochs))
        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for batch in tqdm(train_dataloader):
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


def mix_model_test(shallow_model, new_model, test_dataloader, device, model_type,
                   new_corpus_id):
    new_model.eval()
    old_skiping, old_skipmodel, new_skiping, new_skipmodel = shallow_model

    pred_token_list = []
    label_token_list = []

    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        input_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

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

        b_preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
        b_label_ids = labels.cpu().numpy().tolist()
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

        modelConfig = BertConfig.from_pretrained(new_model_path)
        modelConfig.has_adapter = use_adapter
        modelConfig.isLinear = False
        modelConfig.num_labels = len(unique_tags)

        new_model = SkipBertAdapterForTokenClassification.from_pretrained(new_model_path, config=modelConfig)

        new_model.freeze_model(True)  # freeze all params
        # unfreeze adapter params
        adapter_param = ["adapter_fi", "adapter_se"]
        adapter_param_list = [p for n, p in new_model.named_parameters() if any(nd in n for nd in adapter_param)]
        for param in adapter_param_list:
            param.requires_grad = True
    else:

        method.setup_seed(seed)

        modelConfig = BertConfig.from_pretrained(new_model_path)
        modelConfig.num_labels = len(unique_tags)

        new_model = SkipBertForTokenClassification.from_pretrained(new_model_path, config=modelConfig)

    new_model.freeze_shallow_layers()
    new_model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, new_model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    train_dataloader = data_load(new_model_path, batch_size, 'train')
    validation_dataloader = data_load(new_model_path, batch_size, 'dev')
    test_dataloader = data_load(new_model_path, batch_size, 'test')
    val_acc = [] 
    test_acc = []

    for name, param in new_model.named_parameters():
        if param.requires_grad:
            print(name)
    model_test(new_model, validation_dataloader, device, 'Val ')
    for i in range(epochs):
        print("Epochs:%d/%d" % ((i + 1), epochs))
        new_model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for batch in tqdm(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

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

            optimizer.zero_grad()
            loss = new_model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels,
                             trigram_states=mix_tri_states)[0]
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

            nb_tr_steps += 1

        scheduler.step()
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        acc = mix_model_test(shallow_model, new_model, validation_dataloader, device, 'Val', new_corpus_id)
        val_acc.append(acc)
        acc1 = mix_model_test(shallow_model, new_model, test_dataloader, device, 'Test', new_corpus_id)
        test_acc.append(acc1)

    return val_acc, test_acc


def main():
    epochs = 15
    batch_size = 2
    learning_rate = 7e-4
    seed = 40
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("-------------------------")
    print(batch_size)
    print(learning_rate)
    print(seed)
    print(device)
    print("--------------------------")

    domain = 'Computer'

    trigram_path_3 = './data/trigram/' + domain + '/trigram_0.3.txt'
    new_corpus_id_3 = []
    with open(trigram_path_3, 'r') as record_file:
        for x in tqdm(record_file.readlines()):
            y = x.strip()
            new_corpus_id_3.append(y)
    new_corpus_id_3 = np.array(new_corpus_id_3)

    trigram_path_5 = './data/trigram/' + domain + '/trigram_0.5.txt'
    new_corpus_id_5 = []
    with open(trigram_path_5, 'r') as record_file:
        for x in tqdm(record_file.readlines()):
            y = x.strip()
            new_corpus_id_5.append(y)
    new_corpus_id_5 = np.array(new_corpus_id_5)

    bert_model0 = './model/skip-mlm-new/'  
    bert_model1 = './model/' + domain + '/' 

    print("Adapter Fre 50% Trianing")
    val_acc3, test_acc3 = mix_fine_tune(bert_model0, bert_model1, learning_rate, epochs,
                                        device, batch_size, seed, new_corpus_id_5, use_adapter=True)

    # print("Adapter Mix 1 Training")
    # val_acc3, test_acc3 = mix_fine_tune(bert_model0, bert_model1, learning_rate,
    #                                     epochs, device, batch_size, seed, -1, use_adapter=True)

    # print("Mix 0.9 Training")
    # val_acc4, test_acc4 = mix_fine_tune(bert_model0, bert_model1, learning_rate,
    #                                     epochs, device, batch_size, seed, 0.0176)

    legend = ['Adapter Fre 50% ']
    Val_Acc = [val_acc3]
    Test_Acc = [test_acc3]

    # method.plot_res('Val Acc', legend, Val_Acc)
    # method.plot_res('Test Acc', legend, Test_Acc)
    method.get_test_acc(legend, Val_Acc, Test_Acc)
    print("--------------------")
    for i in range(len(legend)):
        print(legend[i], end=" ")
        print("Max Test Acc: ")
        print(max(Test_Acc[i]))


if __name__ == "__main__":
    main()
