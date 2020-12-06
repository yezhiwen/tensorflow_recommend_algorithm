# coding=utf-8
"""
-------------------------------------
author : yezhiwen.buaa
introduction : 
-------------------------------------
"""

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_criteo_data():
    data = pd.read_csv('../data/criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    return data[sparse_features + dense_features], data['label']


def input_fn():

    # 加载原始数据
    data, label = load_criteo_data()
    features = dict([(key, tf.convert_to_tensor(data[key].values))for key in data.keys() if key != 'label'])
    label = tf.convert_to_tensor(label.values)

    tensor_dict = {'features': features, 'label': label}

    dataset = tf.data.Dataset.from_tensor_slices(tensor_dict)

    dataset = dataset.batch(64).repeat()
    return dataset


if __name__ == '__main__':

    dataset = input_fn()
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            tmp = sess.run([one_element])
            print(tmp)