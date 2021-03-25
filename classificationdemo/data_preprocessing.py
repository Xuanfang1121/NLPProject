# -*- coding: utf-8 -*-
# @Time    : 2021/3/23 12:02
# @Author  : zxf
import os


def get_classification_data(file_path):
    data = []
    for sub_file in os.listdir(file_path):
        sub_filedir = os.path.join(file_path, sub_file)
        print("sub filedir: ", sub_filedir)
        for file in os.listdir(sub_filedir):
            file = os.path.join(sub_filedir, file)
            with open(file, "r", encoding="utf-8") as f:
                temp = []
                for line in f.readlines():
                    line = line.strip().split()
                    temp.append(''.join(line))
                if ''.join(temp) != '':
                    context = "{}\t{}".format(sub_file, ''.join(temp))
                    data.append(context)

    return data


def save_data(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for context in data:
            f.write(context + "\n")


def filter_null_context(file):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            label, context = line.strip().split(',')
            if context is not None:
                temp = "{},{}".format(label, context)
                data.append(temp)
    return data


def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])


def read_data_file(file_path, output_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(stringQ2B(line))

    with open(output_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line)
    print("数据全角转半角finish")


if __name__ == "__main__":
    # file_path = "D:/spyder/data/sogominidata/"
    # data = get_classification_data(file_path)
    # save_file_path = "./data/sougo_mini_data.txt"
    # save_data(data, save_file_path)
    file_path = "./data/sougo_mini_data.txt"
    output_path = "./data/sougo_mini_data_final.txt"
    read_data_file(file_path, output_path)
    # data = filter_null_context(file_path)
