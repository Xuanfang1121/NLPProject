# -*- coding: utf-8 -*-
# @Time    : 2021/3/15 23:22
# @Author  : zxf
import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


# 多标签分类模型
class BertMultiClassifier(object):
    def __init__(self, bert_model_name, label_num):
        self.label_num = label_num
        self.bert_model_name = bert_model_name

    def get_model(self):
        bert = TFBertModel.from_pretrained(self.bert_model_name)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(None,), dtype=tf.int32, name="attention_mask")

        outputs = bert(input_ids, attention_mask=attention_mask)[1]
        cla_outputs = Dense(self.label_num, activation='sigmoid')(outputs)
        model = Model(
            inputs=[input_ids, attention_mask],
            outputs=[cla_outputs])
        return model