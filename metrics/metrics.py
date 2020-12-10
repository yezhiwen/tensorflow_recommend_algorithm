#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import logging
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


class BinaryMetricHook(tf.estimator.SessionRunHook):
    """
    加入Prefix标识，识别是train的指标，还是eval的指标
    """
    def __init__(self,
                 pred_tensor=None,
                 label_tensor=None,
                 batch_size = 32,
                 prefix='train'):

        self.pred_tensor = pred_tensor
        self.label_tensor = label_tensor
        self.cur_preds = []
        self.cur_labels = []
        self.global_preds = []
        self.global_labels = []
        self.batch_size = batch_size
        self.prefix = prefix
        self.step = 0

    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs([self.pred_tensor, self.label_tensor]) # , tf.estimator.SessionRunArgs(self.label_tensor)

    def after_run(self, run_context, run_values):
        pred, label = run_values.results
        self.cur_preds.extend(pred)
        self.cur_labels.extend(label)

        if len(self.cur_preds) >= self.batch_size:
            self.step += 1

            # print([(pred, label) for pred, label in zip(self.cur_preds, self.cur_labels)])

            # 计算每个batch的指标
            auc, ks, bias, pos_cnt, neg_cnt = self.get_auc(self.cur_labels, self.cur_preds)
            matrix, acc, recall = self.get_confusion_matrix(self.cur_labels, self.cur_preds)

            # 打印日志
            print(
                "{prefix}:[{auc_step}] -> auc:{auc}, ks:{ks}, bias:{bias}, pos_cnt:{pos_cnt}, neg_cnt:{neg_cnt}, acc:{acc}, recall:{recall}"
                .format(prefix=self.prefix, auc_step=self.step, auc=auc, ks=ks, bias=bias, pos_cnt=pos_cnt,
                        neg_cnt=neg_cnt, acc=acc, recall=recall))

            self.global_preds.extend(self.cur_preds)
            self.global_labels.extend(self.cur_labels)
            self.cur_preds = []
            self.cur_labels = []

    # 计算auc指标
    def get_auc(self, labels, scores):
        fpr, tpr, thresholds = roc_curve(y_score=scores, y_true=labels, pos_label=1)
        ks = max(tpr - fpr)
        auc = roc_auc_score(y_score=scores, y_true=labels)

        # cal bias
        scores_sum = np.sum(scores)
        labels_sum = np.sum(labels)
        bias = (scores_sum - labels_sum) / labels_sum
        pos_cnt = labels_sum
        neg_cnt = len(scores) - pos_cnt
        return round(auc, 4), round(ks, 4), round(bias, 4), pos_cnt, neg_cnt

    # 计算准确率指标
    def get_confusion_matrix(self, labels, scores):
        avg = np.average(scores)
        pres = [1 if score >= avg else 0 for score in scores]
        matrix = confusion_matrix(labels, pres)

        if (matrix[0][0] + matrix[0][1]) != 0:
            acc = round(float(matrix[0][0]) / (matrix[0][0] + matrix[0][1]), 4)
        else:
            acc = 0

        if (matrix[0][0] + matrix[1][0]) != 0:
            recall = round(float(matrix[0][0]) / (matrix[0][0] + matrix[1][0]), 4)
        else:
            recall = 0
        return matrix, acc, recall


    def summary(self, scores, labels):

        auc, ks, bias, pos_cnt, neg_cnt = self.get_auc(labels, scores)
        matrix, acc, recall = self.get_confusion_matrix(labels, scores)
        labels = np.array(labels)
        scores = np.array(scores)
        label_avg = np.average(labels)
        predict_avg = np.average(scores)
        pos_predict_avg = np.sum(np.where(labels != 0, scores, 0)) / np.sum(np.where(labels != 0, 1, 0))
        neg_predict_avg = np.sum(np.where(labels != 0, 0, scores)) / np.sum(np.where(labels != 0, 0, 1))

        print(
            "{prefix}:[Summary] -> auc:{auc}, ks:{ks}, bias:{bias}, pos_cnt:{pos_cnt}, neg_cnt:{neg_cnt}, acc:{acc}, recall:{recall}, label_avg:{label_avg}, predict_avg:{predict_avg}, pos_predict_avg:{pos_predict_avg}, neg_predict_avg:{neg_predict_avg}"
                .format(prefix=self.prefix, auc=auc, ks=ks, bias=bias, pos_cnt=pos_cnt, neg_cnt=neg_cnt, acc=acc,
                        recall=recall,
                        label_avg=label_avg, predict_avg=predict_avg, pos_predict_avg=pos_predict_avg,
                        neg_predict_avg=neg_predict_avg))

    def summary_group_by_scores(self, all_scores, all_labels):

        data = {}
        for score, label in zip(all_scores, all_labels):
            key = int(score * 10)
            if not key in data:
                data[key] = []

            data[key].append((score, label))

        data = sorted(data.items(), key=lambda x:x[0], reverse=False)
        for key, value in data:
            scores = [each[0] for each in value]
            labels = [each[1] for each in value]

            if len(scores) <= 0 or len(labels) <= 0 or sum(labels) == len(labels) or sum(labels) == 0  or len(labels) != len(scores):
                print("group:{key} bad data !".format(key=key))
                continue

            label_avg = np.average(labels)
            predict_avg = np.average(scores)
            # print("np.sum(np.where(labels != 0, 1, 0))",np.sum(np.where(labels != 0, 1, 0)))
            pos_predict_avg = np.sum(np.where(np.array(labels) != 0, np.array(scores), 0)) / np.sum(np.where(np.array(labels) != 0, 1, 0))
            neg_predict_avg = np.sum(np.where(np.array(labels) != 0, 0, np.array(scores))) / np.sum(np.where(np.array(labels) != 0, 0, 1))

            auc, ks, bias, pos_cnt, neg_cnt = self.get_auc(labels, scores)
            matrix, acc, recall = self.get_confusion_matrix(labels, scores)
            print(
                "{prefix}:[key:{key}] -> auc:{auc}, ks:{ks}, bias:{bias}, pos_cnt:{pos_cnt}, neg_cnt:{neg_cnt}, acc:{acc}, recall:{recall}, label_avg:{label_avg}, predict_avg:{predict_avg}, pos_predict_avg:{pos_predict_avg}, neg_predict_avg:{neg_predict_avg}"
                    .format(prefix=self._prefix, key=key, auc=auc, ks=ks, bias=bias, pos_cnt=pos_cnt, neg_cnt=neg_cnt, acc=acc,
                            recall=recall,
                            label_avg=label_avg, predict_avg=predict_avg, pos_predict_avg=pos_predict_avg,
                            neg_predict_avg=neg_predict_avg))

    def end(self, session):
        if 'Eval' in self.prefix or 'eval' in self.prefix:
            self.summary(self.global_preds, self.global_labels)