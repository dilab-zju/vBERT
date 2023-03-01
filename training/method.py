#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/5/10 9:29
# software: PyCharm
import json
import os
import random
import re

import datetime
import time

import natsort
import pytz as pytz
import numpy as np
import pdfplumber
import spacy
import tokenizers
from matplotlib import pyplot as plt
from tokenizers.implementations import ByteLevelBPETokenizer
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, BertTokenizerFast, LineByLineTextDataset
from transformers import WEIGHTS_NAME, CONFIG_NAME
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


def clean_pdf():
    sentences = []
    with pdfplumber.open("papers/test01.pdf") as pdf:
        for page in pdf.pages:  
            text = page.extract_text()
            text = re.sub(r'[0-9]+', '', text) 
            text = text.strip('\n')  
            text = text.split('.')  
            sentences.extend(text)

    for sentence in sentences:
        if len(sentence) < 20:
            sentences.pop(sentences.index(sentence))
        else:
            sentence = sentence.replace('\n', '')  
            print(sentence)
            print("----------------")


def read_tsv(path):
    word_list = []
    line_num = 0
    with open(path, 'r') as f:
        for line in f:
            line_num += 1
            text = line.split()
            if int(text[2]) < 300:  
                continue
            else:
                word = text[0]
                if word.isalpha():
                    word_list.append(word)
                else:
                    continue

    for word in word_list:
        print(word)
    print(len(word_list))
    print(line_num)
    return word_list


def read_vocab(path):
    """
    :param path: 
    :return:
    """
    vocab_list = []
    with open(path, 'r') as f:
        print("reading vocab......")
        for vocab in tqdm(f):
            vocab_list.append(vocab)

    return vocab_list


def check_token(ori_vocab, new_vocab):
    """
    :param ori_vocab: 
    :param new_vocab: 
    """
    ori_token = read_vocab(ori_vocab)  
    new_token = read_vocab(new_vocab)
    res = []
    print("comparing token......")
    with open(ori_vocab, 'a') as file:
        for vocab in tqdm(new_token):
            if vocab not in ori_token:
                res.append(vocab)
                file.write(vocab)

    print('New Token Number：%d' % len(res))


def read_json(path):
    datas = []
    print("cleaning json------")
    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines()):
            dic = json.loads(line)
            # print(dic.keys())
            datas.append(dic)
    return datas


def label2num(path):
    """
    :param path: 
    :return:
    """
    label_num = {}
    token_num = 0  
    files = os.listdir(path) 
    for file in files: 
        position = path + file  
        with open(position, 'r', encoding='utf-8') as doc:
            for line in tqdm(doc.readlines()):
                dic = json.loads(line)
                lab = dic['label']
                if lab in label_num.keys():
                    continue
                else:
                    label_num.update({lab: token_num})
                    token_num += 1
    return label_num


def ner_label(path):
    tag_list = []
    files = natsort.natsorted(os.listdir(path), alg=natsort.ns.PATH) 
    for file in files:  
        position = path + file  
        with open(position, 'r', encoding='utf-8') as doc:
            for line in tqdm(doc.readlines()):
                if 'DOCSTART' in line:
                    continue
                else:
                    if len(line) == 1:
                        continue
                    else:
                        tmp = line.split()
                        tag_list.append(tmp[-1])

   
    unique_tags = list(set(tag_list))
    unique_tags = sorted(unique_tags) 
    tag2id = {tag: tag_id for tag_id, tag in enumerate(unique_tags)}
    id2tag = {tag_id: tag for tag, tag_id in tag2id.items()}

    return unique_tags, tag2id, id2tag


def create_text(in_path, out_path):
    """
    
    :param in_path: 
    :param out_path: 
    """
    with open(out_path, "a") as f:
        nlp = spacy.load("en_core_sci_sm")
        print("Cleaning Paper......")
        with open(in_path, 'r', encoding='utf-8') as papers:
            for line in tqdm(papers.readlines()):
                paper = json.loads(line)
                abstract_text = paper['abstract'] 
                body_text = paper['body_text']  
                for abstract in abstract_text:  
                    text0 = abstract['text']
                    doc = nlp(text0)
                    list_text = list(doc.sents)  
                    for sentence in list_text: 
                        f.write(str(sentence) + '\n')
                for body in body_text:
                    text1 = body['text']
                    doc = nlp(text1)
                    list_text = list(doc.sents)
                    for sentence in list_text:
                        f.write(str(sentence) + '\n')


def train_token(filepath, save_path):
    """
    :param filepath: 
    :param save_path: 
    """
    bwpt = tokenizers.BertWordPieceTokenizer()

    bwpt.train(
        files=filepath,
        vocab_size=30000, 
        min_frequency=10,
        limit_alphabet=1000
    )
    
    bwpt.save_model(save_path)

    tokenizer = BertTokenizer(vocab_file=save_path + 'vocab.txt')

    sequence0 = "Setrotech is a part of brain"
    tokens0 = tokenizer.tokenize(sequence0)
    print(tokens0)

    # v_size = len(tokenizer.vocab)  
    # print(v_size)
    # model = BertForMaskedLM.from_pretrained("./Bert/bert-base-uncased")
    # model.resize_token_embeddings(len(tokenizer))


def add_token(path):
    model = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model, use_fast=True)
    model = BertForMaskedLM.from_pretrained(model)

    sequence0 = "Setrotech is a part of brain"
    tokens0 = tokenizer.tokenize(sequence0)
    print(tokens0)

    word_list = read_tsv(path)
    for word in tqdm(word_list):
        tokenizer.add_tokens(word)

    model.resize_token_embeddings(len(tokenizer))

    tokens1 = tokenizer.tokenize(sequence0)
    print(tokens1)

    tokenizer.save_pretrained("Pretrained_LMs/bert-base-cased")


def plot_res(title, legend, datas):
    for data in datas:
        x = np.arange(len(data))
        plt.plot(x, data)
        # plt.ylim(0.76, 0.80)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(legend)
    plt.title(title)
    plt.show()


def time_beijing():
    """
    :return: now Beijing time
    """
    tz = pytz.timezone('Asia/Shanghai')  # 
    t = datetime.datetime.fromtimestamp(int(time.time()), tz).strftime('%Y-%m-%d %H:%M:%S')

    return t


def create_dataloader(bert_model, file_path, device, batch_size):
    sentencses = []

    with open(file_path, 'r', encoding='utf-8') as file:
        print("Reading Data-----")
        for line in tqdm(file.readlines()):
            sent = '[CLS] ' + line + ' [SEP]'  
            sentencses.append(sent)

    tokenizer = BertTokenizerFast.from_pretrained(bert_model, do_lower_case=True)
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]

    MAX_LEN = 128

    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    inputs = torch.tensor(input_ids).to(device)
    masks = torch.tensor(attention_masks).to(device)


    data = TensorDataset(inputs, masks)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


def get_test_acc(legend, val_acc, test_acc):
    for i, model_type in enumerate(legend):
        print(model_type, end='\t')
        index = val_acc[i].index(max(val_acc[i]))
        print('Highest Val Acc:%f,Epoch:%d' % (val_acc[i][index], index), end=', \t')
        print('Corresponding Test Acc:%f' % test_acc[i][index])


def freeze_lower_layers(model, config):
    for p in model.bert.embeddings.parameters():
        p.requires_grad = False
    for layer in model.bert.encoder.layer[
                 :config.num_hidden_layers - config.num_full_hidden_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    try:
        for p in model.bert.shallow_skipping.linear.parameters():
            p.requires_grad = False
    except Exception as e:
        pass
    try:
        for p in model.bert.attn.parameters():
            p.requires_grad = False
    except Exception as e:
        pass

    model.bert.embeddings.dropout.p = 0.
    for layer in model.bert.encoder.layer[
                 :config.num_hidden_layers - config.num_full_hidden_layers]:
        for m in layer.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.

    return model


def freeze_higher_layers(model, config):
    for layer in model.bert.encoder.layer[-config.num_full_hidden_layers:]:
        for p in layer.parameters():
            p.requires_grad = False
        for m in layer.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.

    return model


def feature_base(model):
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def pre_data(tokenizer, filepath):
    print("Loading pretraining data------")
    T1 = time.time()
    train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=filepath, block_size=128)
    T2 = time.time()
    print('Loading data Spended:%.2f s' % (T2 - T1))
    print(time_beijing())
    print("---------------------------------------")

    return train_dataset


def model_save(model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(epochs, model, optimizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

    torch.save(epochs, output_dir + "epochs.pth")
    torch.save(optimizer.state_dict(), output_dir + "optimizer.pth")


def get_subword_id(vocab_path):
    word_id = 0
    word_list = []

    with open(vocab_path, 'r', encoding='utf-8') as file:
        print("Geting Vocab------")
        for line in tqdm(file.readlines()):
            if "#" in line:
                word_list.append(word_id)
            word_id += 1

    return word_list


def mae_dist(a, b):
    minus = b - a
    abs_minus = torch.abs(minus)

    mae = abs_minus.sum().item() / abs_minus.numel()

    return mae


def merge_trigram_states_by_mae(old_trigram_states, new_trigram_states, mae_threshold):
    t1, t2 = old_trigram_states, new_trigram_states
    x, y, z = t1.size()

    minus_t = torch.abs(t2 - t1)
    sum_t = torch.sum(minus_t, dim=(1, 2))
    mae_t = sum_t / (y * z)

    t1_choose = torch.where(mae_t > mae_threshold, 0, 1)
    t2_choose = torch.where(mae_t > mae_threshold, 1, 0)

    t1_choose = t1_choose.unsqueeze(1)
    t1_choose.expand((len(t1), t1.size(1)))
    t1_choose = t1_choose.unsqueeze(2)
    t1_choose.expand(t1.size())

    t2_choose = t2_choose.unsqueeze(1)
    t2_choose.expand((len(t2), t2.size(1)))
    t2_choose = t2_choose.unsqueeze(2)
    t2_choose.expand(t2.size())

    mix_t = t1_choose * t1 + t2_choose * t2

    return mix_t


def input_ids_to_tri_grams(x: np.array):
    bs, seq_len = x.shape
    ret = np.zeros((bs * (seq_len + 1), 3), dtype=np.int64)
    i_ret = 0
    for i_bs in range(bs):
        for i_token in range(seq_len):
            if x[i_bs, i_token] == 0:
                break
            if i_token == 0:
                ret[i_ret][1] = x[i_bs, i_token]
                ret[i_ret][2] = x[i_bs, i_token + 1]
            elif i_token == seq_len - 1:
                ret[i_ret][0] = x[i_bs, i_token - 1]
                ret[i_ret][1] = x[i_bs, i_token]
            else:
                ret[i_ret] = x[i_bs, i_token - 1:i_token + 2]
            i_ret += 1
        i_ret += 1  # add a pad trigram between seqs
    return ret[:i_ret]


def id2string(trigrams_id):
    res = []
    for i in range(len(trigrams_id)):
        x = trigrams_id[i]
        id_string = str(x[0]) + " " + str(x[1]) + " " + str(x[2])
        res.append(id_string)

    return np.array(res)


def merge_trigram_states_by_input_id(old_trigram_states, new_trigram_states, input_id, new_corpus_id):
    device = new_trigram_states.device

    t1, t2 = old_trigram_states, new_trigram_states

    t1_choose = np.isin(input_id, new_corpus_id, invert=True)
    t2_choose = np.isin(input_id, new_corpus_id)

    t1_choose = t1_choose + 0
    t2_choose = t2_choose + 0

    t1_choose = torch.from_numpy(t1_choose).to(device)
    t2_choose = torch.from_numpy(t2_choose).to(device)

    t1_choose = t1_choose.unsqueeze(1)
    t1_choose.expand((len(t1), t1.size(1)))
    t1_choose = t1_choose.unsqueeze(2)
    t1_choose.expand(t1.size())

    t2_choose = t2_choose.unsqueeze(1)
    t2_choose.expand((len(t2), t2.size(1)))
    t2_choose = t2_choose.unsqueeze(2)
    t2_choose.expand(t2.size())

    mix_t = t1_choose * t1 + t2_choose * t2

    return mix_t


def layer_save(model, save_path):
    model_to_save = model.module if hasattr(model, 'module') else model
    param_list = {}
    for name, param in model_to_save.named_parameters():
        if param.requires_grad:
            param_list.update({name: param})

    torch.save(param_list, save_path)
