# -*- coding: utf-8 -*-
# @Time    : 2021/3/25 11:40
# @Author  : zxf
import os
import json
import requests

import numpy as np
from flask import Flask
from flask import jsonify
from flask import request

from utils import load_dict
from utils import get_tokenizer
from config.classification_config import params

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

args = params()
max_len = args["max_len"]
tag2id = load_dict(args["tag2id"])
id2tag = {value: key for key, value in tag2id.items()}

tokenizer = get_tokenizer()


def create_infer_inputs(sentences, max_len, tokenizer):
    input_ids_list = []
    attention_mask_list = []
    input_ids = []
    for idx, word in enumerate(sentences):
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
    input_ids_list.append(input_ids)
    attention_mask_list.append(attention_mask)
    return input_ids_list, attention_mask_list


@app.route("/classification", methods=['POST'])
def classification_predict():
    data = json.loads(request.get_data(), encoding="utf-8")
    sentence = data["context"]

    input_ids, attention_mask = create_infer_inputs(sentence, max_len, tokenizer)
    print("input_ids: ", input_ids)
    print("attention_mask: ", attention_mask)
    data = json.dumps({"signature_name": "serving_default",
                       "inputs": {"input_ids": input_ids,
                                  "attention_mask": attention_mask}})
    headers = {"content-type": "application/json"}
    result = requests.post("http://192.168.4.193:8009/v1/models/class:predict", data=data, headers=headers)
    print("result: ", result)
    if result.status_code == 200:
        result = json.loads(result.text)
        logits = np.array(result["outputs"])
        pred = np.argmax(logits, axis=1).tolist()
        pred_label = id2tag[pred[0]]
        print(pred_label)
        return_result = {"code": 200,
                         "context": sentence,
                         "label": pred_label}
        return jsonify(return_result)
    else:
        return_result = {"code": 200,
                         "context": sentence,
                         "label": None}
        return jsonify(return_result)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)