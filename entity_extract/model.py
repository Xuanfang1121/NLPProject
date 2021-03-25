# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 20:00
# @Author  : zxf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFBertModel


def create_model(num_tags, max_len):
    # # BERT encoder
    encoder = TFBertModel.from_pretrained("bert-base-chinese")
    # save_path = "D:/spyder/tf_torch_model/torch_model/bert_base_uncased/"
    # save_path = "/work/zhangxf/torch_pretraining_model/bert_base_uncased/"
    save_path = "D:/Spyder/pretrain_model/transformers_torch_tf/bert_base_chinese/"
    encoder.save_pretrained(save_path)

    # # NER Model
    input_ids = layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    token_type_ids = layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids")
    attention_mask = layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]
    embedding = layers.Dropout(0.3)(embedding)

    tag_logits = layers.Dense(num_tags, activation='softmax')(embedding)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[tag_logits],
    )
    optimizer = keras.optimizers.Adam(lr=3e-5)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )

    def masked_ce_loss(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 17))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)
    model.compile(optimizer=optimizer, loss=masked_ce_loss, metrics=['accuracy'])
    return model