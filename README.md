# Managing Flock of BERTs in the Cloud for Cost-Effective Inference

# About the project

> This project aims to tackle the challenge of efficiently serving massive numbers of customized pre-trained language models (PLM) in a single commodity cloud server while maintaining high throughput and cost-effectiveness. To accomplish this, we propose to combine two techniques: lower layer skipping and parameter-efficient fine-tuning. Additionally, we introduce virtualized model processors to serve large numbers of tenants, resulting in cost-effective PLM serving without sacrificing inference speed or accuracy.
> The system is composed of three key components: a dispatcher, a manager, and a scheduler, which work together to handle user requests, manage PLM instances, and arrange and optimize inference pipelines for hardware efficiency. We conducted extensive experiments across multiple scenarios and workloads and discovered that the system was capable of serving up to 10K vBERT instances on a single GPU with a reasonable inference speed and only a minimal decrease in accuracy.

# Quick Start

## Dependencies

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge cupy nccl cudatoolkit=11.6
conda install numba
pip install transformers==4.12.5
```

## Training

Please refer to `./training` folder.

## Serving

### Pipelined Inference

Pipeline optimization is a technique to make the deep learning model's inference faster. It involves overlapping the time overhead of different parts of the model to reduce idle time of hardware like GPUs. 

In this project, we apply this on two places: 
(1) Retrieving ngram representations and transformer computation; 
(2) Loading adapter parameters while processing the main network. 
These techniques can speed up the inference by up to 25%.

We use CUDA stream to implement the above optimization. Please refer to `vbert/modeling_async.py` for details.

## Sending Requests

TODO


If you find this code useful in your research, please consider citing:
```bib
@inproceedings{wang2023smile,
  title={SMILE: A Cost-Effective System for Serving Massive Pretrained Language Models in the Cloud (Demo)},
  author={Wang, Jue and Chen, Ke and Shou, Lidan and Jiang, Dawei and Chen, Gang},
  booktitle={Proceedings of the 2023 International Conference on Management of Data},
  year={2023}
}
```
