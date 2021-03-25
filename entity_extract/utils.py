# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 19:20
# @Author  : zxf
import os
import json

from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer


def get_tokenizer():
    # Save the slow pretrained tokenizer
    slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  #
    # save_path = "D:/spyder/tf_torch_model/torch_model/bert_base_uncased/"
    # save_path = "/work/zhangxf/torch_pretraining_model/bert_base_uncased/"
    save_path = "D:/Spyder/pretrain_model/transformers_torch_tf/bert_base_uncased/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

    # Load the fast tokenizer from saved file
    # "bert_base_uncased/vocab.txt"
    # tokenizer_file = "/work/zhangxf/torch_pretraining_model/bert_base_uncased/vocab.txt"
    tokenizer_file = "D:/Spyder/pretrain_model/transformers_torch_tf/bert_base_uncased/vocab.txt"
    tokenizer = BertWordPieceTokenizer(tokenizer_file,
                                       lowercase=True)
    return tokenizer


def load_data(data_file, tag2id):
    data = []
    label = []
    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    temp_data = []
    temp_label = []
    for line in lines:
        index = len(tag2id)
        if line == "\n":
            data.append(temp_data)
            label.append(temp_label)
            temp_data = []
            temp_label = []
        else:
            word, tag = line.strip().split()
            if tag not in tag2id:
                tag2id[tag] = index
            temp_data.append(word)
            temp_label.append(tag)
    return data, label, tag2id


def label_encoder(label, tag2id):
    label_encoder = []
    for iterm in label:
        temp = [tag2id[word] for word in iterm]
        label_encoder.append(temp)
    return label_encoder


def save_dict(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))


def load_dict(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)