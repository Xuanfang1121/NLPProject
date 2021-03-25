# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 20:24
# @Author  : zxf
import os
import json
import requests

import numpy as np
from flask import Flask, request, jsonify

from utils import load_dict
from utils import get_tokenizer
from predict import create_infer_inputs

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

max_len = 100
tag2id = load_dict("./output/tag2id.json")
id2tag = {value: key for key, value in tag2id.items()}
tokenizer = get_tokenizer()


@app.route('/', methods=['POST'])
def bert_ner_infer():
    params = json.loads(request.get_data(), encoding="utf-8")
    text = params["text"]
    url = params["url"]
    x, len_list = create_infer_inputs(text, max_len, tokenizer)
    print("len_list: ", len_list)
    input_ids = x[0].tolist()
    token_type_ids = x[1].tolist()
    attention_mask = x[2].tolist()
    data = json.dumps({"signature_name": "serving_default",
                       "inputs": {"input_ids": input_ids,
                                  "token_type_ids": token_type_ids,
                                  "attention_mask": attention_mask}})
    headers = {"content-type": "application/json"}
    result = requests.post(url, data=data, headers=headers)
    result = json.loads(result.text)
    pred_logits = result["outputs"][0]
    pred = np.argmax(pred_logits, axis=1).tolist()
    print("pred: ", pred)
    # todo 感觉写的有问题，用训练集的样本预测都是O
    predict_label = []
    for j in range(min(len_list[0], max_len)):
        predict_label.append(id2tag[pred[j]])

    return_result = {"predict": predict_label}
    return jsonify(return_result)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
