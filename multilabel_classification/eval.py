# -*- coding: utf-8 -*-
# @Time    : 2021/3/17 13:47
# @Author  : zxf
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from tensorflow.keras.callbacks import Callback


class Metrics(Callback):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    # def on_train_begin(self, logs={}):
    #     self.val_f1s = []
    #     self.val_recalls = []
    #     self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.x))).round()
        val_targ = self.y
        _val_f1 = f1_score(val_targ, val_predict, average="macro",
                           zero_division=0)
        _val_recall = recall_score(val_targ, val_predict, average="macro",
                                   zero_division=0)
        _val_precision = precision_score(val_targ, val_predict, average="macro",
                                         zero_division=0)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('-val_f1: %.4f --val_precision: %.4f --val_recall: %.4f' % (_val_f1, _val_precision, _val_recall))
        return
