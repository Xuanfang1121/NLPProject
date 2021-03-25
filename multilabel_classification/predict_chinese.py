# -*- coding: utf-8 -*-
# @Time    : 2021/3/19 14:28
# @Author  : zxf
import os

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import read_dict
from config import get_args
from utils import create_model
from utils import get_tokenizer


# 预测data整理
def get_model_data(data, tokenizer, max_seq_len=128):
    dataset_dict = {
        "input_ids": [],
        "attention_mask": [],
    }

    for sentence in data:
        input_ids = tokenizer.encode(
            sentence,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_seq_len,  # Truncate all sentences.
        )
        sentence_length = len(input_ids)
        input_ids = pad_sequences([input_ids],
                                  maxlen=max_seq_len,
                                  dtype="long",
                                  value=0,
                                  truncating="post",
                                  padding="post")
        input_ids = input_ids.tolist()[0]
        attention_mask = [1] * sentence_length + [0] * (max_seq_len - sentence_length)

        dataset_dict["input_ids"].append(input_ids)
        # dataset_dict["token_type_ids"].append(token_type_ids)
        dataset_dict["attention_mask"].append(attention_mask)

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["attention_mask"],
    ]
    return x


def main(test_data, args, label_num):
    # test_steps_per_epoch = len(test_data) // args["batch_size"]
    tokenizer = get_tokenizer(args['bert_model_name'], args['pretrain_model_path'])
    testdata = get_model_data(test_data, tokenizer, args["max_length"])
    print("testdata: ", testdata)
    model = create_model(args['bert_model_name'], label_num)
    model.load_weights("./output/model/mulclassifition.h5")

    pred_logits = model.predict(testdata, batch_size=args["batch_size"])
    pred = np.where(pred_logits >= 0.5, 1, 0).tolist()
    # pred = np.where(pred < 0.5, pred, 1).tolist()
    return pred


def label_encoder(pred, label):
    result = []
    for iterm in pred:
        index = [i for i in range(len(iterm)) if iterm[i] == 1]
        if len(index) == 0:
            result.append(['unk'])
        elif len(index) == 1:
            result.append([label[index[0]]])
        else:
            temp = [label[i] for i in index]
            result.append(['|'.join(temp)])
    return result


if __name__ == "__main__":
    args = get_args()
    df_test = pd.read_csv(args["test_file"])
    test_data = df_test["content"].tolist()
    label = read_dict(args["labeldict"])["label"]
    pred = main(test_data, args, len(label))
    print("pred: ", pred[:2])
    pred_encoder = label_encoder(pred, label)
    print("label encoder: ", pred_encoder[:3])
    pred_df = pd.DataFrame(pred_encoder, columns=["pred"])
    df_test = pd.concat([df_test, pred_df], axis=1) #df_test.append(pred_df, ignore_index=True)
    df_test = df_test[["label", "pred", "content"]]
    df_test.to_csv("./output/test_predict.csv", index=False, header=True, encoding="utf-8-sig")

    # test_data = ["昨天18：30，陕西宁强县胡家坝镇向家沟村三组发生山体坍塌，5人被埋。当晚，3人被救出，其中1人在医院抢救无效死亡，"
    #              "2人在送医途中死亡。今天凌晨，另外2人被发现，已无生命迹象。"]
    # pred = main(test_data, args, len(label))
    # print("pred: ", pred)
    # pred_encoder = label_encoder(pred, label)
    # print("pred label: ", pred_encoder)