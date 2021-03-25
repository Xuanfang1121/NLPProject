# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 11:41
# @Author  : zxf
import os

import numpy as np
import tensorflow as tf
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report

from utils import load_data
from utils import save_dict
from model import create_model
from utils import label_encoder
from utils import get_tokenizer


def create_inputs_targets(sentences, tags, tag2id, max_len, tokenizer):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "tags": []
    }

    for sentence, tag in zip(sentences, tags):
        input_ids = []
        target_tags = []
        for idx, word in enumerate(sentence):
            ids = tokenizer.encode(word, add_special_tokens=False)
            input_ids.extend(ids.ids)
            # 这个判断ids的长度会避免很多错误，tokenizer中会出现多个值，对应的label也要相加，例如对一个韩文token后会出现多个值
            num_tokens = len(ids)
            target_tags.extend([tag[idx]] * num_tokens)

        # Pad truncate，句子前后加'[CLS]','[SEP]'
        input_ids = input_ids[:max_len - 2]
        target_tags = target_tags[:max_len - 2]

        input_ids = [101] + input_ids + [102]
        # 这里'O'对应的是16, 这里是否对应的是tag2id中的[CLS][SEP]
        target_tags = [tag2id['O']] + target_tags + [tag2id['O']]
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        padding_len = max_len - len(input_ids)
        # vocab中 [PAD]的编码是0
        input_ids = input_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        # target 这里新加一个label是应该是对应[SEP]或者[CLS],或者是'O'
        # taget padding 'O'
        target_tags = target_tags + ([tag2id['O']] * padding_len)

        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["token_type_ids"].append(token_type_ids)
        dataset_dict["attention_mask"].append(attention_mask)
        dataset_dict["tags"].append(target_tags)
        assert len(target_tags) == max_len, f'{len(input_ids)}, {len(target_tags)}'

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = dataset_dict["tags"]
    return x, y


def main():
    train_file = "./data/train_example.txt"
    dev_file = "./data/dev_example.txt"
    tag2id_path = "./output/tag2id.json"
    output_path = "./output/"
    pb_path = "./output/1"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.join(pb_path):
        os.makedirs(pb_path)
    tag2id = dict()
    max_len = 64
    batch_size = 4
    epoch = 1
    # load data
    train_data, train_label, tag2id = load_data(train_file, tag2id)
    print("train data size: ", len(train_data))
    print("train label size: ", len(train_label))
    print("label dict: ", tag2id)
    dev_data, dev_label, tag2id = load_data(dev_file, tag2id)
    print("dev data size: ", len(dev_data))
    print("dev label size: ", len(dev_label))
    print("label dict: ", tag2id)
    # save tag2id
    save_dict(tag2id, tag2id_path)
    # label encoder
    train_label = label_encoder(train_label, tag2id)
    print("train label: ", train_label[:3])
    dev_label = label_encoder(dev_label, tag2id)
    print("dev label: ", dev_label[:3])
    # get tokenizer
    tokenizer = get_tokenizer()
    # 准备模型数据
    train_x, train_y = create_inputs_targets(train_data, train_label, tag2id, max_len, tokenizer)
    print("train data tokenizer: ", train_x[:3])
    dev_x, dev_y = create_inputs_targets(dev_data, dev_label, tag2id, max_len, tokenizer)
    print("dev data tokenizer: ", dev_x[:3])

    # create model
    model = create_model(len(tag2id), max_len)
    model.summary()
    history = model.fit(train_x,
                        train_y,
                        epochs=epoch,
                        verbose=1,
                        batch_size=batch_size,
                        validation_data=(dev_x, dev_y),
                        validation_batch_size=batch_size
                        )   # , validation_split=0.1

    # model save
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model_file = os.path.join(output_path, "ner_model.h5")
    model.save_weights(model_file, overwrite=True)

    # save pb model
    tf.keras.models.save_model(model, pb_path, save_format="tf")

    # 模型评价，通过seqeval
    pred = model.predict(train_x, batch_size=batch_size)
    print("pred shape: ", pred.shape)

    # 可视化模型训练的loss和accuracy
    # fig1 = plt.figure()
    # plt.plot(history.history['loss'], 'r', linewidth=3.0)
    # plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    # plt.legend(['train loss', 'val loss'])
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.title('loss curve', fontsize=16)
    # fig1.savefig("./data/loss.png")
    # plt.show()
    #
    # fig2 = plt.figure()
    # plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    # plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
    # plt.legend(['train acc', 'val acc'])
    # plt.xlabel('epoch')
    # plt.ylabel('acc')
    # fig2.savefig("./data/acc.png")
    # plt.show()


if __name__ == "__main__":
    main()