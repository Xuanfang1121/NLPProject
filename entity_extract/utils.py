# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 19:20
# @Author  : zxf
import os
import json

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer

from model import BertNERModel


def get_tokenizer(model_path):
    # Save the slow pretrained tokenizer
    slow_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")  #
    # save_path = "D:/spyder/tf_torch_model/torch_model/bert_base_chinese/"
    # save_path = "/work/zhangxf/torch_pretraining_model/bert_base_chinese/"
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # slow_tokenizer.save_pretrained(save_path)

    # Load the fast tokenizer from saved file
    # "bert_base_uncased/vocab.txt"
    vocab_file = os.path.join(model_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file,
                                       lowercase=True)
    return tokenizer


def load_data(data_file):
    data = []
    label = []
    tag2id = dict()
    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    temp_data = []
    temp_label = []
    length_list = []
    for line in lines:
        index = len(tag2id)
        if line == "\n":
            data.append(temp_data)
            label.append(temp_label)
            length_list.append(len(temp_label))
            temp_data = []
            temp_label = []
        else:
            word, tag = line.strip().split()
            if tag not in tag2id:
                tag2id[tag] = index
            temp_data.append(word)
            temp_label.append(tag)
    return data, label, tag2id, length_list


def label_encoder(label, tag2id):
    label_encoder = []
    for iterm in label:
        temp = [tag2id[word] for word in iterm]
        label_encoder.append(temp)
    return label_encoder


def save_dict(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))


def load_dict(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_model(model_path, num_tags, dropout):
    model = BertNERModel(model_path, num_tags, dropout).get_model()

    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    train_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )

    def masked_ce_loss(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 17))
        loss_ = train_loss(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    model.compile(optimizer=optimizer, loss=masked_ce_loss, metrics=['accuracy'])
    return model


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


def create_infer_inputs(sentences, max_len, tokenizer):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": []
    }
    len_list = []
    for sent in sentences:
        input_ids = []
        for word in sent:
            ids = tokenizer.encode(word, add_special_tokens=True)
            input_ids.extend(ids.ids)
        len_list.append(len(input_ids))

        # input_ids
        input_ids = input_ids[:max_len - 2]
        input_ids = [101] + input_ids + [102]
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        # padding
        padding_len = max_len - len(input_ids)
        input_ids = input_ids + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)

        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["token_type_ids"].append(token_type_ids)
        dataset_dict["attention_mask"].append(attention_mask)

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [dataset_dict["input_ids"], dataset_dict["token_type_ids"], dataset_dict["attention_mask"]]
    return x, len_list


def bio_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    iCount = 0
    entity_tag = ""
    # assert len(string)==len(tags), "string length is: {}, tags length is: {}".format(len(string), len(tags))

    for c_idx in range(len(tags)):
        c, tag = string[c_idx], tags[c_idx]
        if c_idx < len(tags)-1:
            tag_next = tags[c_idx+1]
        else:
            tag_next = ''

        if tag[0] == 'B':
            entity_tag = tag[2:]
            entity_name = c
            entity_start = iCount
            if tag_next[2:] != entity_tag:
                # item["entities"].append({"word": c, "start": iCount, "end": iCount + 1, "type": tag[2:]})
                item["entities"].append({"name": c, "index": iCount, "tag": tag[2:]})
        elif tag[0] == "I":
            if tag[2:] != tags[c_idx-1][2:] or tags[c_idx-1][2:] == 'O':
                tags[c_idx] = 'O'
                pass
            else:
                entity_name = entity_name + c
                if tag_next[2:] != entity_tag:
                    # item["entities"].append({"word": entity_name, "start": entity_start, "end": iCount + 1,
                    # "type": entity_tag})
                    item["entities"].append({"name": entity_name, "index": entity_start,
                                             "tag": entity_tag})
                    entity_name = ''
        iCount += 1
    return item