# coding=utf-8
"""
-------------------------------------
author : yezhiwen.buaa
introduction : 
-------------------------------------
"""
import tensorflow as tf

def get_nn_layers(input_layer, dims, activate_fuc='relu', dropout=0.1):

    layer = input_layer
    for units in dims[0:-1]:
        layer = tf.keras.layers.Dense(units,
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=tf.keras.regularizers.l1(1e-4))(layer)
        # layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(activate_fuc)(layer)
        layer = tf.keras.layers.Dropout(dropout)(layer)

    layer = tf.keras.layers.Dense(dims[-1],
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=tf.keras.regularizers.l1(1e-4))(layer)

    return layer


def get_logits(sparse_feature, dense_feature):
    pass