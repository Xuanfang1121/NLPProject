# -*- coding: utf-8 -*-
# @Time    : 2021/3/25 10:39
# @Author  : zxf
import os
import json

import numpy as np
import tensorflow as tf

from utils import load_dict
from utils import load_data
from utils import create_model
from utils import get_tokenizer
from common.common import logger
from utils import create_infer_inputs
from config.classification_config import params
"""
   1. 数据特征处理，模型预测的特征
   2. 加载模型
   3. 模型预测
"""


def text_classifier_predict(sentences, max_len, tag2id, bert_model_name, model_path):
    # get tokenizer
    tokenizer = get_tokenizer()
    test_x = create_infer_inputs(sentences, max_len, tokenizer)
    # id2tag
    id2tag = {value: key for key, value in tag2id.items()}
    # model
    model = create_model(bert_model_name, len(tag2id))
    model.load_weights(model_path)
    logits = model.predict(test_x)
    pred = np.argmax(logits, axis=1).tolist()
    pred_label = [id2tag[i] for i in pred]
    print("preict label: ", pred_label)


if __name__ == "__main__":
    args = params()
    tag2id = load_dict(args["tag2id"])
    test_data, _ = load_data(args["test_file"], tag2id)
    print("test_data: ", test_data[:3])
    model_path = os.path.join(args["output_path"], "classification_model.h5")
    text_classifier_predict(test_data[:3], args["max_len"], tag2id, args["bert_model_name"], model_path)
