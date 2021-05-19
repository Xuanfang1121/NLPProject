# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 19:19
# @Author  : zxf
import numpy as np

from utils import load_data
from utils import create_model
from utils import get_tokenizer
from utils import create_infer_inputs
from config.ner_config import model_params


def model_predict():
    max_len = 128
    args = model_params()
    test_data, test_label, _, _ = load_data(args["test_file"])
    print("test data size: ", len(test_data))
    tokenizer = get_tokenizer(args["pretrain_model_path"])
    test_x, len_list = create_infer_inputs(test_data, max_len, tokenizer)
    print("test data tokenizer: ", test_x[:3])
    tag2id = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6}
    model = create_model(args["pretrain_model_path"], len(tag2id), args["dropout"])
    model.load_weights("./output/ner_model.h5")
    pred_logits = model.predict(test_x)
    id2tag = {value: key for key, value in tag2id.items()}
    # shape [batch_size, seq_len]
    pred = np.argmax(pred_logits, axis=2).tolist()
    predict_label = []
    for i in range(len(len_list)):
        temp = []
        temp_pred = pred[i]
        for j in range(min(len_list[i], max_len)):
            temp.append(id2tag[temp_pred[j]])
        predict_label.append(temp)
    print("predict label: ", predict_label)


if __name__ == "__main__":
    model_predict()
