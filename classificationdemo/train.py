# -*- coding: utf-8 -*-
# @Time    : 2021/3/10 10:57
# @Author  : zxf
import os
import json

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import load_data
from utils import save_dict
from utils import create_model
from utils import label_encoder
from utils import get_tokenizer
from utils import random_shuffle
from common.common import logger
# from evaluate import model_evaluate
from utils import create_inputs_targets
from config.classification_config import params


def main():
    args = params()
    tag2id_path = os.path.join(args["output_path"], args["tag2id"])

    if not os.path.exists(args["output_path"]):
        os.makedirs(args["output_path"])
    if not os.path.join(args["pb_path"]):
        os.makedirs(args["pb_path"])
    tag2id = {"体育": 0, "健康": 1, "军事": 2, "教育": 3, "汽车": 4}
    max_len = args["max_len"]
    batch_size = args["batch_size"]
    epoch = args["epoch"]
    # load data
    data, label = load_data(args["data_file"], tag2id)
    logger.info("total data size: {}".format(len(data)))
    logger.info("total label size: {}".format(len(label)))
    # random 乱序
    data, label = random_shuffle(data, label)
    # save tag2id
    save_dict(tag2id, tag2id_path)
    # label encoder
    total_label = label_encoder(label, len(tag2id))

    # get train test data
    train_data, dev_data, train_label, dev_label = train_test_split(data, total_label, test_size=0.2)
    logger.info("train data size: {}".format(len(train_data)))
    logger.info("dev data size: {}".format(len(dev_data)))
    # bert tokenizer
    tokenizer = get_tokenizer()
    # tokenizer = get_roberta_tokenizer()
    # 准备模型数据
    train_x, train_y = create_inputs_targets(train_data, train_label, max_len, tokenizer)
    dev_x, dev_y = create_inputs_targets(dev_data, dev_label, max_len, tokenizer)

    # create model bert
    # model = create_model(len(tag2id))
    model = create_model(args["bert_model_name"], len(tag2id))
    # model.summary()
    model.fit(train_x,
              train_y,
              epochs=epoch,
              verbose=1,
              batch_size=batch_size,
              validation_data=(dev_x, dev_y),
              validation_batch_size=batch_size
              )   # , validation_split=0.1

    # model save
    model_path = os.path.join(args["output_path"], "classification_model.h5")
    model.save_weights(model_path, overwrite=True)

    # save pb model
    tf.keras.models.save_model(model, args["pb_path"], save_format="tf", overwrite=True)


if __name__ == "__main__":
    main()