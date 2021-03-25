# -*- coding: utf-8 -*-
# @Time    : 2021/3/19 11:07
# @Author  : zxf
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from config import get_args
from utils import read_dict
from utils import get_tokenizer
from utils import get_model_data
from utils import create_model
from utils import label_encoder


"""
   训练主函数
"""


def main():
    args = get_args()
    train_df = pd.read_csv(args["train_file"])
    train_df = shuffle(train_df)
    train_datas = train_df["content"].tolist()

    train_label_total = train_df["label"].unique().tolist()
    print("total data size: {}".format(len(train_datas)))
    # get lable dict
    label_list = read_dict(args["labeldict"])["label"]
    if not os.path.exists(args["labeldict"]):
        for label in train_label_total:
            if "|" in label:
                temp = label.split("|")
                for item in temp:
                    if item not in label_list:
                        label_list.append(item)
            else:
                if label not in label_list:
                    label_list.append(label)
        print("label cate size: {}".format(len(label_list)))
        label_dict = {"label": label_list}
        with open(args["labeldict"], "w", encoding="utf-8") as f:
            f.write(json.dumps(label_dict, ensure_ascii=False, indent=4))

    # label encoder
    train_labels = label_encoder(train_df["label"].tolist(), label_list)

    train_data, val_data, train_label, val_label = train_test_split(train_datas,
                                                                    train_labels,
                                                                    test_size=0.2,
                                                                    random_state=0)
    print("train data size: {}".format(len(train_data)))
    print("val data size: {}".format(len(val_data)))

    tokenizer = get_tokenizer(args["bert_model_name"],
                              args["pretrain_model_path"])

    train_x, train_y = get_model_data(train_data, train_label, tokenizer,
                                      args["max_length"])

    val_x, val_y = get_model_data(val_data, val_label, tokenizer, args["max_length"])
    model = create_model(args["bert_model_name"], len(label_list))

    if not os.path.exists(args["model_path"]):
        os.makedirs(args["model_path"])

    if not os.path.exists(args["pbmodel_path"]):
        os.makedirs(args["pbmodel_path"])

    # 设置保存最优的模型,保存的是pb模型
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            # The saved model name will include the current epoch.
            filepath=args["model_path"],  # {epoch}
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor='val_auc',  # 'accuracy',
            verbose=1,
            mode='max'
        )
    ]

    model.fit(train_x, train_y, epochs=args["epoch"], verbose=1,
              batch_size=args["batch_size"],
              callbacks=callbacks,
              validation_data=(val_x, val_y),
              validation_batch_size=args["batch_size"])

    model_path = os.path.join("./output/model/", "mulclassifition.h5")
    model.save_weights(model_path)

    tf.keras.models.save_model(model, args["pbmodel_path"], save_format="tf", overwrite=True)


if __name__ == "__main__":
    main()
