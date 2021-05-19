# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 20:00
# @Author  : zxf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFBertModel
from transformers import TFBertForTokenClassification


class BertNERModel(object):
    def __init__(self, model_path, num_tags, dropout):
        self.model_path = model_path
        self.num_tags = num_tags
        self.dropout = dropout
        self.encoder = TFBertForTokenClassification.from_pretrained(self.model_path)

    def get_model(self):
        input_ids = layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
        token_type_ids = layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids")
        attention_mask = layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
        embedding = self.encoder(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )[0]
        embedding = layers.Dropout(self.dropout)(embedding)

        tag_logits = layers.Dense(self.num_tags, activation='softmax')(embedding)

        model = keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask],
            outputs=[tag_logits],
        )
        return model