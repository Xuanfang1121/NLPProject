# -*- coding: utf-8 -*-
# @Time    : 2021/2/3 15:15
# @Author  : zxf
import numpy as np
from seqeval.metrics import f1_score
from seqeval.metrics import recall_score
from seqeval.metrics import precision_score
from seqeval.metrics import classification_report

from common.common import logger

"""
    模型评价，计算precision， recall， f1_score, classification_report
"""


def model_evaluate(model, data, label, tag2id, batch_size, seq_len_list):
    id2tag = {value: key for key, value in tag2id.items()}
    pred_logits = model.predict(data, batch_size=batch_size)
    # pred shape [batch_size, max_len]
    preds = np.argmax(pred_logits, axis=2).tolist()

    assert len(preds) == len(seq_len_list)
    # get predcit label
    predict_label = []
    target_label = []
    for i in range(len(preds)):
        pred = preds[i][1:]
        temp = []
        true_label = label[i][:min(seq_len_list[i], len(pred))]
        for j in range(min(seq_len_list[i], len(pred))):
            temp.append(id2tag[pred[j]])
        assert len(temp) == len(true_label)
        target_label.append(true_label)
        predict_label.append(temp)

    # 计算 precision， recall， f1_score
    precision = precision_score(target_label, predict_label, average="macro", zero_division=0)
    recall = recall_score(target_label, predict_label, average="macro", zero_division=0)
    f1 = f1_score(target_label, predict_label, average="macro", zero_division=0)
    logger.info(classification_report(target_label, predict_label))
    return precision, recall, f1


def model_evaluate_roberta(model, data, label, tag2id, batch_size, seq_len_list):
    id2tag = {value: key for key, value in tag2id.items()}
    pred_logits = model.predict(data, batch_size=batch_size)[0]
    # pred shape [batch_size, max_len]
    preds = np.argmax(pred_logits, axis=2).tolist()

    assert len(preds) == len(seq_len_list)
    # get predcit label
    predict_label = []
    target_label = []
    for i in range(len(preds)):
        pred = preds[i][1:]
        temp = []
        true_label = label[i][:min(seq_len_list[i], len(pred))]
        for j in range(min(seq_len_list[i], len(pred))):
            temp.append(id2tag[pred[j]])
        assert len(temp) == len(true_label)
        target_label.append(true_label)
        predict_label.append(temp)

    # 计算 precision， recall， f1_score
    precision = precision_score(target_label, predict_label, average="macro",
                                zero_division=0)
    recall = recall_score(target_label, predict_label, average="macro",
                          zero_division=0)
    f1 = f1_score(target_label, predict_label, average="macro",
                  zero_division=0)
    logger.info(classification_report(target_label, predict_label))
    return precision, recall, f1