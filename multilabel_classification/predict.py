# -*- coding: utf-8 -*-
# @Time    : 2021/3/17 0:39
# @Author  : zxf
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import read_dict
from config import get_args
from utils import get_tokenizer
from utils import create_model


def get_model_data(data, tokenizer, max_seq_len=128):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
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
        dataset_dict["attention_mask"].append(attention_mask)

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["attention_mask"],
    ]
    return x


def main():
    args = get_args()
    df_test = pd.read_csv(args["test_file"])
    test_data = df_test['comment_text'].values.tolist()
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    tokenizer = get_tokenizer(args['bert_model_name'], args['pretrain_model_path'])
    testdata = get_model_data(test_data, tokenizer, args["max_length"])

    model = create_model(args["bert_model_name"], len(label_cols))

    model.load_weights(args["model_path"])
    pred_logits = model.predict(testdata)
    pred = np.where(pred_logits > 0.15, 1, 0).tolist()
    print(pred)


if __name__ == "__main__":
    main()