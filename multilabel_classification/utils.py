# -*- coding: utf-8 -*-
# @Time    : 2021/3/15 23:27
# @Author  : zxf
import os
import json

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import BertMultiClassifier


# bert tokenizer
def get_tokenizer(bert_model_name, save_path):
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
    # save_path = "/work/zhangxf/torch_pretraining_model/bert_base_uncased/"
    # save_path = "D:/Spyder/pretrain_model/transformers_torch_tf/bert_base_uncased/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)
    return tokenizer


def get_model_data(data, labels, tokenizer, max_seq_len=128):
    dataset_dict = {
        "input_ids": [],
        "attention_mask": [],
        "label": []
    }
    assert len(data) == len(labels)
    for i in range(len(data)):
        sentence = data[i]
        input_ids = tokenizer.encode(
            sentence,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_seq_len,  # Truncate all sentences.
        )
        sentence_length = len(input_ids)
        input_ids = pad_sequences([input_ids],
                                  maxlen=max_seq_len,
                                  dtype="long",
                                  value=0,
                                  truncating="post",
                                  padding="post")
        input_ids = input_ids.tolist()[0]
        # token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * sentence_length + [0] * (max_seq_len - sentence_length)

        dataset_dict["input_ids"].append(input_ids)
        # dataset_dict["token_type_ids"].append(token_type_ids)
        dataset_dict["attention_mask"].append(attention_mask)
        dataset_dict["label"].append(labels[i])

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        # dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = dataset_dict["label"]
    return x, y


def create_model(bert_model_name, label_nums):
    # model = BertClassifier(TFBertModel.from_pretrained(bert_model_name), label_nums)
    model = BertMultiClassifier(bert_model_name, label_nums).get_model()
    optimizer = tf.keras.optimizers.Adam(lr=1e-5)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validation_loss = tf.keras.metrics.Mean(name='test_loss')
    model.compile(optimizer=optimizer, loss=loss_object,
                  metrics=['accuracy', tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC()])   # metrics=['accuracy']
    return model


def label_encoder(y, label_list):
    """
    :param y: label data, list type
    :param label_list: label 字典
    :return:
    """
    labels = []
    for iterm in y:
        temp = iterm.split("|")
        label = [0] * len(label_list)
        if len(temp) > 0:
            for item in temp:
                index = label_list.index(item)
                label[index] = 1
        labels.append(label)
    return labels


# read dict file
def read_dict(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data