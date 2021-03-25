# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 19:19
# @Author  : zxf
import os
import json

import numpy as np
import tensorflow as tf

from utils import load_data
from utils import load_dict
from model import create_model
from utils import get_tokenizer


def create_infer_inputs(sentences, max_len, tokenizer):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": []
    }
    len_list = []
    for sent in sentences:
        input_ids = []
        for word in sent:
            ids = tokenizer.encode(word, add_special_tokens=False)
            input_ids.extend(ids.ids)
        len_list.append(len(input_ids))

        # input_ids
        input_ids = input_ids[:max_len - 2]
        input_ids = [101] + input_ids + [102]
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        # padding
        padding_len = max_len - len(input_ids)
        input_ids = input_ids + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)

        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["token_type_ids"].append(token_type_ids)
        dataset_dict["attention_mask"].append(attention_mask)

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [dataset_dict["input_ids"], dataset_dict["token_type_ids"],
         dataset_dict["attention_mask"]]
    return x, len_list


def predict(test_data, max_len, tag2id):
    tokenizer = get_tokenizer()
    test_x, len_list = create_infer_inputs(test_data, max_len, tokenizer)
    print("test data tokenizer: ", test_x[:3])
    model = create_model(len(tag2id), max_len)
    model.load_weights("./output/ner_model.h5")
    pred_logits = model.predict(test_x)
    id2tag = {value: key for key, value in tag2id.items()}
    # shape [batch_size, seq_len]
    pred = np.argmax(pred_logits, axis=2).tolist()
    predict_label = []
    for i in range(len(len_list)):
        temp = []
        temp_pred = pred[i]
        for j in range(min(len_list[i], max_len)):
            temp.append(id2tag[temp_pred[j]])
        predict_label.append(temp)
    print("predict label: ", predict_label)
    return predict_label


if __name__ == "__main__":
    test_file = "./data/test_example.txt"
    dict_file = "./output/tag2id.json"
    tag2id = {}
    max_len = 128
    test_data, test_label, _ = load_data(test_file, tag2id)
    print("test data : ", test_data)
    tag2id = load_dict(dict_file)
    pred = predict(test_data, max_len, tag2id)