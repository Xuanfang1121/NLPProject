# -*- coding: utf-8 -*-
# @Time    : 2021/3/17 1:06
# @Author  : zxf
import os
import json
import requests
import traceback

import numpy as np
from flask import request
from flask import Flask, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import get_args
from utils import get_tokenizer


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

args = get_args()
tokenizer = get_tokenizer(args['bert_model_name'], args['pretrain_model_path'])


def get_model_data(sentence, tokenizer, max_seq_len=128):
    dataset_dict = {
        "input_ids": [],
        "attention_mask": [],
    }

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
    attention_mask = [1] * sentence_length + [0] * (max_seq_len - sentence_length)

    dataset_dict["input_ids"].append(input_ids)
    dataset_dict["attention_mask"].append(attention_mask)

    # for key in dataset_dict:
    #     dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["attention_mask"],
    ]
    return x


def label_encoder(pred, label):
    result = []
    for iterm in pred:
        index = [i for i in range(len(iterm)) if iterm[i] == 1]
        if len(index) == 0:
            result.append(['unk'])
        elif len(index) == 1:
            result.append([label[index[0]]])
        else:
            temp = [label[i] for i in index]
            result.append(['|'.join(temp)])
    return result


def read_dict(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


label = read_dict(args["labeldict"])["label"]


@app.route("/multiclassfier", methods=['POST'])
def multiclassifier_pred():
    data_para = json.loads(request.get_data(), encoding="utf-8")
    sentence = data_para["sent"]
    print("sentence: ", sentence)
    url = data_para.get("url")
    # get model input
    test_x = get_model_data(sentence, tokenizer, 256)
    input_ids = test_x[0]
    attention_mask = test_x[1]
    data = json.dumps({"signature_name": "serving_default",
                       "inputs": {"input_ids": input_ids,
                                  "attention_mask": attention_mask}})
    headers = {"content-type": "application/json"}
    result = requests.post(url, data=data, headers=headers)
    if result.status_code == 200:
        result = json.loads(result.text)
        pred_logits = np.array(result["outputs"])
        pred = np.where(pred_logits >= 0.5, 1, 0).tolist()
        pred_encoder = label_encoder(pred, label)
        return_result = {"code": 200, "sent": sentence, "label": pred_encoder[0]}
        return jsonify(return_result)
    else:
        return jsonify({"code": result.status_code,
                        "message": traceback.format_exc()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
