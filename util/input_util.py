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
from sklearn.model_selection import train_test_split
from structs.input_structs import SparseFeat, DenseFeat


def load_criteo_data(path, sep=','):
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
        SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4) for i, feat in enumerate(sparse_features)
    ]
    dense_feature_columns = [DenseFeat(feat, 1,) for feat in dense_features]

    return data[sparse_features], data[dense_features], sparse_feature_columns, dense_feature_columns, data['label']


def input_fn_back(sparse_features, dense_features, label, batch_size=64, shuffle_size=1000, epoch=1):

    # 加载原始数据
    # sparse_features, dense_features, sparse_feature_columns, dense_feature_columns, label = load_criteo_data(path)

    # 原始数据转tensor
    sparse_features =  dict([(key, tf.convert_to_tensor(sparse_features[key].values))for key in sparse_features.keys()])
    dense_features = dict([(key, tf.convert_to_tensor(dense_features[key].values)) for key in dense_features.keys()])
    label = tf.convert_to_tensor(label.values, dtype=tf.float32)

    # 创建数据集
    tensor_dict = {'sparse_features': sparse_features, 'dense_features': dense_features, 'label': label}
    dataset = tf.data.Dataset.from_tensor_slices(tensor_dict)

    # 设置数据集的batch size、shuffle、epoch
    dataset = dataset.batch(batch_size) # .shuffle(shuffle_size).repeat(epoch)
    iterator = dataset.make_one_shot_iterator()

    # 定义features、labels 并返回
    data = iterator.get_next()
    features = {'sparse_features': data['sparse_features'], 'dense_features': data['dense_features']}
    labels = data['label']
    return features, labels


def input_fn(sparse_features, dense_features, label, batch_size=64, shuffle_size=1000, epoch=1):

    # 加载原始数据
    # sparse_features, dense_features, sparse_feature_columns, dense_feature_columns, label = load_criteo_data(path)

    # 原始数据转tensor
    sparse_features =  dict([(key, tf.convert_to_tensor(sparse_features[key].values))for key in sparse_features.keys()])
    dense_features = dict([(key, tf.convert_to_tensor(dense_features[key].values)) for key in dense_features.keys()])
    label = tf.convert_to_tensor(label.values, dtype=tf.float32)

    def parse_data(data):
        features = {'sparse_features': data['sparse_features'], 'dense_features': data['dense_features']}
        labels = data['label']
        return features, labels

    # 创建数据集
    tensor_dict = {'sparse_features': sparse_features, 'dense_features': dense_features, 'label': label}
    dataset = tf.data.Dataset.from_tensor_slices(tensor_dict).map(parse_data)

    # 设置数据集的batch size、shuffle、epoch
    dataset = dataset.batch(batch_size) # .shuffle(shuffle_size).repeat(epoch)
    # iterator = dataset.make_one_shot_iterator()

    # 定义features、labels 并返回
    # data = iterator.get_next()
    # features = {'sparse_features': data['sparse_features'], 'dense_features': data['dense_features']}
    # labels = data['label']
    return dataset


def test_input_fn():
    path = '../data/criteo_sample.txt'
    sparse_features, dense_features, sparse_feature_columns, dense_feature_columns, label = load_criteo_data(path)
    feature, label = input_fn(sparse_features, dense_features, label)
    C1 = feature['sparse_features']['C1']
    with tf.Session() as sess:
        for i in range(3):
            print(C1)
            tmp = sess.run([C1])
            print(tmp)

if __name__ == '__main__':
    test_input_fn()