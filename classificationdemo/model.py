# -*- coding: utf-8 -*-
# @Time    : 2021/3/10 10:57
# @Author  : zxf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFBertForSequenceClassification, TFBertModel


def create_model(num_tags):
    # # BERT encoder
    # encoder = TFBertForSequenceClassification.from_pretrained(
    #     "D:/spyder/tf_torch_model/torch_model/bert_base_chinese/", num_labels=num_tags)
    encoder = TFBertModel.from_pretrained("D:/spyder/tf_torch_model/torch_model/bert_base_chinese/")
    save_path = "D:/spyder/tf_torch_model/torch_model/bert_base_chinese/"
    # # save_path = "/work/zhangxf/torch_pretraining_model/bert_base_chinese/"
    encoder.save_pretrained(save_path)
    # #
    input_ids = layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    embedding = encoder(input_ids, attention_mask=attention_mask)
    print(embedding)
    output = embedding[1]
    output = layers.Dropout(0.3)(output)

    logits = layers.Dense(num_tags, activation='softmax')(output)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[logits],
    )
    optimizer = keras.optimizers.Adam(lr=3e-5)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC()])
    return model


class BertTextClassifier(object):
    def __init__(self, bert_model_name, label_num):
        self.label_num = label_num
        self.bert_model_name = bert_model_name

    def get_model(self):
        bert = TFBertModel.from_pretrained(self.bert_model_name)
        input_ids = keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
        attention_mask = keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")

        outputs = bert(input_ids, attention_mask=attention_mask)[1]
        cla_outputs = layers.Dense(self.label_num, activation='sigmoid')(outputs)
        model = keras.Model(
            inputs=[input_ids, attention_mask],
            outputs=[cla_outputs])
        return model