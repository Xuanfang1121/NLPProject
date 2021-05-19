# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 11:41
# @Author  : zxf
import os
import json

import numpy as np
import tensorflow as tf

from utils import load_data
from utils import save_dict
from utils import create_model
from utils import label_encoder
from utils import get_tokenizer
from common.common import logger
from evaluate import model_evaluate
from utils import create_inputs_targets
from config.ner_config import model_params


def main():
    args = model_params()
    tag2id_path = os.path.join(args["output_path"], args["tag2id"])

    if not os.path.exists(args["output_path"]):
        os.makedirs(args["output_path"])
    if not os.path.join(args["pb_path"]):
        os.makedirs(args["pb_path"])
    max_len = args["max_len"]
    batch_size = args["batch_size"]
    epoch = args["epoch"]
    # load data
    train_data, train_label_ori, tag2id, train_len = load_data(args["train_file"])
    print("train data size: ", len(train_data))
    print("train label size: ", len(train_label_ori))
    print("label dict: ", tag2id)
    dev_data, dev_label_ori, tag2id, dev_len = load_data(args["dev_file"])
    print("dev data size: ", len(dev_data))
    print("dev label size: ", len(dev_label_ori))
    print("label dict: ", tag2id)
    # load test data

    # save tag2id
    save_dict(tag2id, tag2id_path)
    # label encoder
    train_label = label_encoder(train_label_ori, tag2id)
    print("train label: ", train_label[:3])
    dev_label = label_encoder(dev_label_ori, tag2id)
    print("dev label: ", dev_label[:3])
    # get tokenizer
    # bert tokenizer
    tokenizer = get_tokenizer(args["pretrain_model_path"])
    # tokenizer = get_roberta_tokenizer()
    # 准备模型数据
    train_x, train_y = create_inputs_targets(train_data, train_label, tag2id, max_len, tokenizer)
    dev_x, dev_y = create_inputs_targets(dev_data, dev_label, tag2id, max_len, tokenizer)

    # create model bert
    model = create_model(args["pretrain_model_path"], len(tag2id), args["dropout"])
    model.summary()
    model.fit(train_x,
              train_y,
              epochs=epoch,
              verbose=1,
              batch_size=batch_size,
              validation_data=(dev_x, dev_y),
              validation_batch_size=batch_size
             )   # , validation_split=0.1

    # model save
    model_file = os.path.join(args["output_path"], "ner_model.h5")
    model.save_weights(model_file, overwrite=True)

    # save pb model
    tf.keras.models.save_model(model, args["pb_path"], save_format="tf")

    # 模型评价
    precision, recall, f1 = model_evaluate(model, dev_x, dev_label_ori, tag2id, batch_size, dev_len)
    logger.info("model precision:{} recall:{} f1:{}".format(precision, recall, f1))


if __name__ == "__main__":
    main()