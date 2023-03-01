#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/5/24 9:43
# software: PyCharm
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

import random
import numpy as np
import spacy
import tokenizers
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, RobertaConfig, BertTokenizerFast, get_linear_schedule_with_warmup
from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments

import time
import method
from skipbert.modeling import SkipBertForMaskedLM, TestBertForMaskedLM


def hook(grad):
    grad[:30522] = 0.
    return grad


def bert_embeddings(bert_model):
    model_path = bert_model
    token_path = bert_model
    tokenizer = BertTokenizerFast.from_pretrained(token_path, do_lower_case=True)
    config = BertConfig.from_pretrained(model_path)
    model = SkipBertForMaskedLM.from_pretrained(model_path, config=config)
    model.resize_token_embeddings(len(tokenizer))

    model = method.freeze_higher_layers(model, config)
    # model = method.freeze_lower_layers(model, config)

    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad)


def pre_train(bert_model, save_path, train_dataset, learn_rate):

    model_path = bert_model
    token_path = bert_model
    tokenizer = BertTokenizerFast.from_pretrained(token_path, do_lower_case=True)
    config = BertConfig.from_pretrained(model_path)
    model = SkipBertForMaskedLM.from_pretrained(model_path, config=config)


    model = method.freeze_higher_layers(model, config)
    # model = method.freeze_lower_layers(model, config)

    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad)


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    pretrain_batch_size = 192
    num_train_epochs = 3

    # learn_rate = 6e-5  #

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad]}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learn_rate)

    training_args = TrainingArguments(
        output_dir=save_path, overwrite_output_dir=True,
        num_train_epochs=num_train_epochs, learning_rate=learn_rate,
        per_device_train_batch_size=pretrain_batch_size,
        fp16=True, save_total_limit=2)

    trainer = Trainer(
        model=model, args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        optimizers=(optimizer, None))


    trainer.train()
    trainer.save_model(save_path)


def main():
    model_path = './model/skip-mlm-new/' 
    file_path = './data/datasets/Biology_ori_128.pt'  
    # file_path = './data/datasets/chemprot.pt'  
    model_save = "./outputs/Biology/"
    learn_rate = 5e-5

    train_dataset = torch.load(file_path)
    print(method.time_beijing())
    print("--------------------------------")

    pre_train(model_path, model_save, train_dataset, learn_rate)

    # for x in range(2, 9):
    #     learn_rate = x / (10 ** 5)
    #     model_save = './outputs/learnRateTest/' + str(learn_rate) + '/'
    #     print(model_save)
    #     print("Learning Rate:" + str(learn_rate) + "Training------")
    #     if os.path.exists(model_save):
    #         pre_train(skip_model, model_save, train_dataset, learn_rate, True)
    #     else:
    #         pre_train(skip_model, model_save, train_dataset, learn_rate, False)


if __name__ == '__main__':
    main()
