#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2022/12/1 9:37
# software: PyCharm
import time
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset
from transformers import BertTokenizerFast
import natsort
import method


def clean_data(in_path, out_path):
    # Corpus json to txt
    file_name = natsort.natsorted(os.listdir(in_path), alg=natsort.ns.PATH)
    for file in file_name:
        file_path = in_path + file
        new_name = file_path[:-3]
        os.rename(file_path, new_name)
        method.create_text(new_name, out_path)


def save_dataset(bert_model, file_name):
    file_path = './data/sentence/'
    datasets_path = './data/datasets/'
    tokenizer = BertTokenizerFast.from_pretrained(bert_model, do_lower_case=True)
    print("Loading Data------")
    T1 = time.time()
    train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path + file_name + '.txt', block_size=128)
    T2 = time.time()
    print('Loading Data Cost:%s s' % (T2 - T1))

    print("Saving Data------")
    T3 = time.time()
    torch.save(train_dataset, datasets_path + file_name + '_ori_128.pt')
    T4 = time.time()
    print('Saving data cost:%s s' % (T4 - T3))


def json2dataSet(domain, model_path):
    in_path = './data/S2ORC/' + domain + '/pdf_parses/medicine/'
    out_path = './data/sentence/' + domain + '.txt'

    clean_data(in_path, out_path)

    save_dataset(model_path, domain)


def analysis_trigram(trigram_path, trigram_num, sum_num, alpha):
    ratio = int(alpha * trigram_num)
    records_trigram_num = 0
    records_trigram = []

    with open(trigram_path + 'dic.txt', 'r') as dic_file:
        for x in tqdm(dic_file.readlines()):
            y = x.strip()
            y = y.split(',')
            records_trigram_num += int(y[1])
            if records_trigram_num < ratio:
                records_trigram.append(y[0])
            else:
                break
    with open(trigram_path + 'trigram_' + str(alpha) + '.txt', 'w') as record_file:
        for x in records_trigram:
            record_file.write(x + '\n')

    num = len(records_trigram)
    num_ratio = 100 * num / sum_num
    print('The number of trigrams whose frequency weight accounts for %d %% is %d, and the number ratio is: %.4f%%' % ((10 * alpha), num, num_ratio))
    occupy_size = num * 3 * 768 * 2 / 1024 / 1024 / 1024
    print("Size：%.2f GB" % occupy_size)


def save_trigram(model_path, file_path, trigram_path):
    print("Start Time：")
    print(method.time_beijing())
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True)

    print("Loading Data-------")
    datasets = torch.load(file_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    dataloader = DataLoader(datasets, batch_size=batch_size, collate_fn=data_collator)

    tri_grams_dic = {}
    tri_grams_num = 0

    print("Cleaning Data-------------------")
    with open(trigram_path + 'list.txt', 'w', encoding='utf-8') as list_file:
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)

            tri_grams_array = method.input_ids_to_tri_grams(input_ids.cpu().numpy())
            tri_grams_num += len(tri_grams_array)

            for x in tri_grams_array:
                tri_gram = str(x[0]) + " " + str(x[1]) + " " + str(x[2])
                # tri_gram = str(x)
                if tri_gram in tri_grams_dic.keys():
                    tri_grams_dic.update({tri_gram: tri_grams_dic[tri_gram] + 1})
                else:
                    tri_grams_dic.update({tri_gram: 1})

                list_file.write(tri_gram + "\n")

    print("Recording trigram----------------")
    sort_dic = sorted(tri_grams_dic.items(), key=lambda t: t[1], reverse=True)
    with open(trigram_path + 'dic.txt', 'w') as dic_file:
        for x in sort_dic:
            dic_file.write(str(x[0]) + ',' + str(x[1]) + '\n')

    print("tri_grams number: %d" % tri_grams_num)
    print("non-repeating tri_grams number: %d" % len(sort_dic))

    print("End Time：")
    print(method.time_beijing())

    analysis_trigram(trigram_path, tri_grams_num, len(sort_dic), 0.5)
    analysis_trigram(trigram_path, tri_grams_num, len(sort_dic), 0.3)


def trigram_size():
    datasets_path = './data/datasets/'
    model_path = './model/bert-base-uncased/'
    corpus_path = './data/S2ORC/'
    domain_list = natsort.natsorted(os.listdir(corpus_path), alg=natsort.ns.PATH)

    for domain in domain_list:
        print(domain + "Cleaning!")
        print("-------------------------------")

        json2dataSet(domain, model_path)

        trigram_path = './data/trigram/' + domain + '/'
        os.makedirs(trigram_path, exist_ok=True)

        save_trigram(model_path, datasets_path + domain + '_ori_128.pt', trigram_path)


def main():
    trigram_size()


if __name__ == "__main__":
    main()
