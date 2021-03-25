# -*- coding: utf-8 -*-
# @Time    : 2021/3/10 10:57
# @Author  : zxf
import os
import json

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from transformers import RobertaTokenizer

from model import BertTextClassifier


def get_tokenizer():
    # Save the slow pretrained tokenizer
    slow_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")  #
    save_path = "D:/spyder/tf_torch_model/torch_model/bert_base_chinese/"
    # save_path = "/work/zhangxf/torch_pretraining_model/bert_base_chinese/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

    # Load the fast tokenizer from saved file
    # "bert_base_uncased/vocab.txt"
    vocab_file = os.path.join(save_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file,
                                       lowercase=True)
    return tokenizer


def get_roberta_tokenizer():
    slow_tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")  #
    # save_path = "D:/spyder/tf_torch_model/torch_model/bert_base_chinese/"
    save_path = "/work/zhangxf/torch_pretraining_model/chinese_roberta_wwm_ext/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)
    vocab_file = os.path.join(save_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file,
                                       lowercase=True)
    return tokenizer


def load_data(data_file, label2id):
    """
    :param data_file: 数据集文件，label,sentence
    :return: data, label, type 为list
    """
    data = []
    label = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            temp = line.strip().split('\t')
            if len(temp) == 2:
                data.append([temp[1]])
                label.append([label2id[temp[0]]])
    return data, label


def random_shuffle(data, label):
    """
    :param data: [[]]
    :param label: [[]]
    :return:
    """
    index = [i for i in range(len(data))]
    np.random.shuffle(index)
    data_ = np.array(data)[index].tolist()
    label_ = np.array(label)[index].tolist()
    return data_, label_


def label_encoder(label, label_num):
    label_encoder = []
    for iterm in label:
        temp = [0] * label_num
        temp[iterm[0]] = 1
        label_encoder.append(temp)
    return label_encoder


def save_dict(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))


def load_dict(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_inputs_targets(sentences, labels, max_len, tokenizer):
    dataset_dict = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    assert len(sentences) == len(labels)
    for i in range(len(sentences)):
        input_ids = []
        for idx, word in enumerate(sentences[i]):
            ids = tokenizer.encode(word, add_special_tokens=False)
            input_ids.extend(ids.ids)

        # Pad truncate，句子前后加'[CLS]','[SEP]'
        input_ids = input_ids[:max_len - 2]
        input_ids = [101] + input_ids + [102]
        # 这里'O'对应的是16, 这里是否对应的是tag2id中的[CLS][SEP]
        attention_mask = [1] * len(input_ids)
        padding_len = max_len - len(input_ids)
        # vocab中 [PAD]的编码是0
        input_ids = input_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["attention_mask"].append(attention_mask)
        dataset_dict["labels"].append(labels[i])
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["attention_mask"],
    ]
    y = dataset_dict["labels"]
    return x, y


def create_model(bert_model_name, label_nums):
    # model = BertClassifier(TFBertModel.from_pretrained(bert_model_name), label_nums)
    model = BertTextClassifier(bert_model_name, label_nums).get_model()
    optimizer = tf.keras.optimizers.Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC()])   # metrics=['accuracy']
    return model


def create_infer_inputs(sentences, max_len, tokenizer):
    dataset_dict = {
        "input_ids": [],
        "attention_mask": [],
    }
    for i in range(len(sentences)):
        input_ids = []
        for idx, word in enumerate(sentences[i]):
            ids = tokenizer.encode(word, add_special_tokens=False)
            input_ids.extend(ids.ids)

        # Pad truncate，句子前后加'[CLS]','[SEP]'
        input_ids = input_ids[:max_len - 2]
        input_ids = [101] + input_ids + [102]
        # 这里'O'对应的是16, 这里是否对应的是tag2id中的[CLS][SEP]
        attention_mask = [1] * len(input_ids)
        padding_len = max_len - len(input_ids)
        # vocab中 [PAD]的编码是0
        input_ids = input_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["attention_mask"].append(attention_mask)
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["attention_mask"],
    ]

    return x