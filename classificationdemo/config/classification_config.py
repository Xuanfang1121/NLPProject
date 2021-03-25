# -*- coding: utf-8 -*-
# @Time    : 2021/3/10 11:03
# @Author  : zxf
import os


def params():
    args = dict()
    base_path = "./data/"
    args["data_file"] = os.path.join(base_path, "sougo_mini_data_final_demo.txt")
    args["test_file"] = os.path.join(base_path, "sougo_mini_data_final_demo.txt")
    args["bert_model_name"] = "D:/spyder/tf_torch_model/torch_model/bert_base_chinese/"
    args["tag2id"] = "tag2id.dict"
    args["output_path"] = "./output/"
    args["pb_path"] = "./output/1/"

    args["max_len"] = 128
    args["batch_size"] = 4
    args["epoch"] = 1
    return args