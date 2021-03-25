# -*- coding: utf-8 -*-
# @Time    : 2021/3/15 23:52
# @Author  : zxf
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import get_args
from utils import get_tokenizer
from utils import get_model_data
from utils import create_model
from eval import Metrics


def main():
    args = get_args()
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df_train = pd.read_csv(args["train_file"])
    train_datas = df_train['comment_text'].tolist()
    train_labels = df_train[label_cols].values.tolist()
    print("train data size: ", len(train_datas))
    print("train label size: ", len(train_labels))

    train_data, val_data, train_label, val_label = train_test_split(train_datas,
                                                                    train_labels,
                                                                    test_size=0.2,
                                                                    random_state=0)

    tokenizer = get_tokenizer(args["bert_model_name"],
                              args["pretrain_model_path"])

    train_x, train_y = get_model_data(train_data, train_label, tokenizer,
                                      args["max_length"])

    val_x, val_y = get_model_data(val_data, val_label, tokenizer, args["max_length"])
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    model = create_model(args["bert_model_name"], len(label_cols))

    # 自定义计算f1 score
    # metrics = Metrics(val_x, val_y)
    # callbacks = [metrics]

    # 设置保存最优的模型,保存的是pb模型
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            # The saved model name will include the current epoch.
            filepath="./output/model/1",   # {epoch}
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor='auc',   # 'accuracy',
            verbose=1,
        )
    ]

    model.fit(train_x, train_y, epochs=args["epoch"], verbose=1,
              batch_size=args["batch_size"],
              callbacks=callbacks,
              validation_data=(val_x, val_y),
              validation_batch_size=args["batch_size"])

    if not os.path.exists(args["model_path"]):
        os.makedirs(args["model_path"])

    model.save_weights(args["model_path"])

    if not os.path.exists(args["pbmodel_path"]):
        os.makedirs(args["pbmodel_path"])
    tf.keras.models.save_model(model, args["pbmodel_path"], save_format="tf")


if __name__ == "__main__":
    main()
