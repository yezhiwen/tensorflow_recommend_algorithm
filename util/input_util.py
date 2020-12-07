# coding=utf-8
"""
-------------------------------------
author : yezhiwen.buaa
introduction : input fn
-------------------------------------
"""

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from structs.input_structs import SparseFeat, DenseFeat
import numpy as np


def load_census_income_data(path, sep=','):
    features_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']
    label_names = ['income_50k', 'marital_stat']
    sparse_feature_names = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                           'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                           'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                           'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                           'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                           'vet_question']
    dense_feature_names = [each for each in features_names if each not in label_names and each not in sparse_feature_names]

    data = pd.read_csv(
        path,
        delimiter=sep,
        index_col=None,
        names=features_names
    )

    # 处理离散特征
    for feat in sparse_feature_names:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 处理连续特征
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_feature_names] = mms.fit_transform(data[dense_feature_names])

    # 处理label
    data["income_50k"] = np.where(data.income_50k == ' 50000+.', 1, 0)
    data["marital_stat"] = np.where(data.marital_stat == ' Never married', 0, 1)

    # 封装好特征
    sparse_feature_columns = [
        SparseFeat(feat, vocabulary_size=data[feat].nunique()) for i, feat in enumerate(sparse_feature_names)
    ]
    dense_feature_columns = [DenseFeat(feat) for feat in dense_feature_names]

    return data[sparse_feature_names], data[dense_feature_names], sparse_feature_columns, dense_feature_columns, data[label_names]


def load_criteo_data(path, sep='\t'):
    data = pd.read_csv(path, sep=sep)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    # 处理离散特征
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 处理连续特征
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 封装好特征
    sparse_feature_columns = [
        SparseFeat(feat, vocabulary_size=data[feat].nunique()) for i, feat in enumerate(sparse_features)
    ]
    dense_feature_columns = [DenseFeat(feat) for feat in dense_features]

    return data[sparse_features], data[dense_features], sparse_feature_columns, dense_feature_columns, data['label']

def input_fn(sparse_features, dense_features, label, if_multi_label=False, batch_size=64, shuffle_size=1000, epoch=1):

    # 原始数据转tensor
    sparse_features =  dict([(key, tf.convert_to_tensor(sparse_features[key].values, dtype=tf.float32))for key in sparse_features.keys()])
    dense_features = dict([(key, tf.convert_to_tensor(dense_features[key].values, dtype=tf.float32)) for key in dense_features.keys()])

    # 判断是否是 multi label任务
    if not if_multi_label:
        label = tf.convert_to_tensor(label.values, dtype=tf.float32)
    else:
        label = dict([(key, tf.convert_to_tensor(label[key].values, dtype=tf.float32))for key in label.keys()])

    def parse_data(data):
        features = {'sparse_features': data['sparse_features'], 'dense_features': data['dense_features']}
        labels = data['label']
        return features, labels

    # 创建数据集
    tensor_dict = {'sparse_features': sparse_features, 'dense_features': dense_features, 'label': label}
    dataset = tf.data.Dataset.from_tensor_slices(tensor_dict).map(parse_data)

    # 设置数据集的batch size、shuffle、epoch
    dataset = dataset.batch(batch_size).shuffle(shuffle_size).repeat(epoch)
    return dataset


##################################### 分割线：以下是测试 #####################################

def test_criteo_input():
    path = '../data/criteo_sample.txt'
    sparse_features, dense_features, sparse_feature_columns, dense_feature_columns, label = load_criteo_data(path, sep='\t')
    dataset = input_fn(sparse_features, dense_features, label)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    C1 = features['sparse_features']['C1']
    with tf.Session() as sess:
        for i in range(3):
            print(C1)
            tmp = sess.run([C1])
            print(tmp)

def test_census_income_input():
    path = '../data/census-income.data.gz'
    sparse_features, dense_features, sparse_feature_columns, dense_feature_columns, label = load_census_income_data(path, sep=',')
    dataset = input_fn(sparse_features, dense_features, label, if_multi_label=True)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    marital_stat = labels['income_50k']
    with tf.Session() as sess:
        for i in range(3):
            print(marital_stat)
            tmp = sess.run([marital_stat])
            print(tmp)

if __name__ == '__main__':
    test_census_income_input()