# -*- coding: utf-8 -*-
# @Time    : 2021/3/15 23:26
# @Author  : zxf


def get_args():
    args = dict()
    # data args
    # args["train_file"] = "./data/train_demo.csv"
    # args["test_file"] = "./data/test_demo.csv"
    # args["test_label_file"] = "./data/test_labels_demo.csv"
    args["train_file"] = "./data/train_ch.csv"
    args["test_file"] = "./data/test_ch.csv"
    args["labeldict"] = "./data/label_dict.json"
    # model args
    # args["bert_model_name"] = "bert-base-uncased"
    # args["pretrain_model_path"] = "D:/Spyder/pretrain_model/transformers_torch_tf/bert_base_uncased/"
    args["bert_model_name"] = "/nlp_group/nlp_pretrain_models/bert-base-chinese/"
    args["pretrain_model_path"] = "/work/zhangxf/torch_pretraining_model/bert_base_chinese/"
    args["epoch"] = 5
    args["batch_size"] = 32
    args["max_length"] = 256
    args["model_path"] = "./output/model/1"
    args["pbmodel_path"] = "./output/pbmodel/1/"

    return args
