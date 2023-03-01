## Dependences
1. The requriements are showed on 'requirements.txt', you can download them by run:
```python
pip install requirements.txt
```

## Download Data
1. Use 'download_data.py' to download training corpus frmo S2ORC ('https://github.com/allenai/s2orc'), you can change the domain to you want;
2. Use 'trigram_analysis.py' to clean the download corpus, this file will create training datasets, training sentence and analysis the trigram of each domain corpus, we will use this datasets for further pre-training;
3. All baseline model files are downloaded from huggingface, you can get the specific download link from the corresponding folder, we can only download the 'config.json', 'pytorch_model.bin' and 'vocab.txt', such as:
```
"./model/bert-base-uncased/link.txt"
```
4. The SkipBert model you can download from 'https://github.com/LorrinWWW/SkipBERT';

## Further Pre-training
1. The folder './skipbert/' are used to build skipbert model;
2. Use 'pretraining.py' to do further pre-training for base model on special domain corpus;
3. When finish further pre-training, the results are saved in folder './outputs/', you can move them to folder './model/' for downstream task fine-tune;

## Downstream Task  Fine-tune
1. 'my_bert.py' are used to build adapter on each model;
2. './data/text_classification/' save the downstream task data of CLS, './data/ner/' save the downstream task data of NER, './data/SubTask/' save the subset of the above all task for thousands test;
3. 'cls_train.py' and 'ner_train.py' implement text classification tasks and named entity recognition tasks respectively, you can select the task you want to verify at the beginning of the code;
4. 'mix_cls.py' and 'mix_ner.py' are downstream tasks done using a mix of old and new trigrams, you can adjust the proportion of the mixture in the code.
