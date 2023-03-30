import json
import multiprocessing
from multiprocessing import Process, Queue
import os
import threading
import time
import requests
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from modeling_async import vBertModel
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# print log ob screen
ch = logging.StreamHandler()
# output put log to disk file
fh = logging.FileHandler(filename='./server.log')
formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)

ch.setFormatter(formatter)
logger.addHandler(ch)


file_s = open(PATH_TO_CORPUS_FILE)
lines = file_s.readlines()
device = torch.device('cuda:0')
input_queue = Queue(1000000)
init_signal_queue = Queue(10)

class Manager:
    def __init__(self, bottleneck_size, hidden_size, num_tenants, num_full_hidden_layers):
        super(Manager, self).__init__()
        self.all_adapters = []
        self.bottleneck_size = bottleneck_size
        self.hidden_size = hidden_size
        self.num_tenants = num_tenants
        self.num_full_hidden_layers = num_full_hidden_layers
        self.domain_index = {}
        self.domain_names = ["finance", "bio", "news"]
        
    def load_all_adapters(self):

        count = 0
        for i in range(0, len(self.domain_names)):
            self.domain_index[self.domain_names[i]] = count
            count = count + 1

            domain_adapers = {}
            for i in range(0, self.num_tenants):
                adapter_param = [
                    np.zeros((1, 1 * self.bottleneck_size * self.hidden_size \
                              + 1 * self.bottleneck_size \
                              + 1 * self.hidden_size * self.bottleneck_size \
                              + 1 * self.hidden_size), dtype=np.float32) for layer_i in range(self.num_full_hidden_layers)
                ]
                domain_adapers[i] = adapter_param

            self.all_adapters.append(domain_adapers)

        logger.info("load all adapters over")

        return self.all_adapters


    def find_adapters_by_domain_and_id(self, domain_name, id):
        domain_index = self.domain_index[domain_name]
        return self.all_adapters[domain_index][id]


def vbert_server(input_queue, signal_queue):
    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_TOKENIZER)
    config = AutoConfig.from_pretrained(PATH_TO_CONFIG)
    config.batch_size = 1700
    config.bottleneck_size = 64
    config.num_full_hidden_layers = 6
    config.max_num_entries = 100000
    num_tenants = 10000

    # generate single user's PLOT
    batch_size = 200
    adapter_param = [
        np.zeros((batch_size, 1 * config.bottleneck_size * config.hidden_size \
                  + 1 * config.bottleneck_size \
                  + 1 * config.hidden_size * config.bottleneck_size \
                  + 1 * config.hidden_size), dtype=np.float32) for layer_i in range(config.num_full_hidden_layers)
    ]

    config.plot_mode = 'update_all'
    vbert = vBertModel.from_pretrained(PATH_TO_MODEL, config=config).eval().to(device)
    x = tokenizer(lines[batch_size], padding='max_length', max_length=128, return_tensors='pt', truncation=True).input_ids.to(device)

    with torch.no_grad():
        tmp = vbert(x, adapter_param=adapter_param)


    # load all adapters to host memory
    adapter_manager = Manager(config.bottleneck_size, config.hidden_size, num_tenants, config.num_full_hidden_layers)
    adapter_manager.load_all_adapters()

    signal_queue.put(1)
    # handling inference requests
    config.plot_mode = 'plot_only'
    while True:
        item = input_queue.get()
        logger.info("vbert got inference request from dispatcher")
        id = item['id']
        seq = item['seq']
        domain_name = item['domain_name']

        adapter_param = adapter_manager.find_adapters_by_domain_and_id(domain_name, id)

        x_cpu = tokenizer(seq, padding='max_length', max_length=128, return_tensors='pt', truncation=True).input_ids
        output = vbert(x_cpu, adapter_param=adapter_param)


# next for fastAPI
class Request(BaseModel):
    id: int
    domain_name: str
    seq: str


app = FastAPI()


@app.get("/init_process")
def init():
    # start vbert server
    Process(target=vbert_server, args=(input_queue, init_signal_queue, )).start()
    while True:
        if init_signal_queue.qsize() != 0:
            break
    return "init vbert server done"


def init_model():
    time.sleep(5)
    url = "http://0.0.0.0:9989/init_process"
    res = requests.get(url)
    logger.info(res.text)



@app.post("/inference")
def Dispatcher(request : Request):
    dic = request.dict()
    user_id = dic['id']
    domain_name = dic['domain_name']
    seq = dic['seq']
    input_queue.put(dic)
    return "ok"



if __name__ == '__main__':
    # try:
    #     torch.multiprocessing.set_start_method(method='spawn', force=True)
    # except RuntimeError:
    #     pass
    #

    Process(target=init_model).start()
    uvicorn.run(app="vBert_test_for_github:app", host="0.0.0.0", port=9989, reload=True, workers=4)
