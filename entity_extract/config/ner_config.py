# -*- coding: utf-8 -*-
# @Time    : 2021/2/3 16:24
# @Author  : zxf
import os

from config.global_conf import PROJECT_DIR


def model_params():
    args = dict()
    args["train_file"] = os.path.join(PROJECT_DIR, "data/train.txt")
    args["dev_file"] = os.path.join(PROJECT_DIR, "data/dev.txt")
    args["test_file"] = os.path.join(PROJECT_DIR, "data/test.txt")
    args["output_path"] = os.path.join(PROJECT_DIR, "output/")
    args["tag2id"] = "tag2id.json"
    args["pb_path"] = os.path.join(PROJECT_DIR, "output/1/")

    args["pretrain_model_path"] = "D:/spyder/pretrain_model/transformers_torch_tf/chinese_roberta_wwm_ext/"
    # model params
    args["epoch"] = 5
    args["batch_size"] = 4
    args["max_len"] = 128
    args["dropout"] = 0.3

    return args