# coding=utf-8
"""
-------------------------------------
author : yezhiwen.buaa
introduction : Wide & Deep
-------------------------------------
"""


import tensorflow as tf
from layer import common_layer, layers
from metrics import metrics
from util import model_util

def model_fn(features, labels, mode, params):

    """
    :param features: input_fn 得到的feature
    :param labels: labels
    :param mode: 当前模式：分为训练、验证、预测
    :param params: 模型的其他参数
    :return: estimator(推算器)
    """

    """
        输入feature & 配置的读取
    """

    # 获取 feature tensor
    # 1. 对于 sparse_feature 存储的是index
    # 2. 对于 dense_feature 存储的是具体的值
    sparse_features_input = features['sparse_features']
    dense_features_input = features['dense_features']

    # 获取基本配置
    batch_size = params['batch_size']
    lr = params['lr']
    embedding_size = params['embedding_size']

    # 获取需要进入 linear和nn部分的feature name
    linear_feature_names = params['linear_feature_names']
    dnn_feature_names = params['dnn_feature_names']

    # 获取 sparse、dense feature 配置
    sparse_feature_columns = params['sparse_feature_columns']

    # 创建 sparse feature embedding
    sparse_embeddings = model_util.get_feature_embeddings(
        sparse_feature_columns,
        dict([(feature_name, input) for feature_name, input in sparse_features_input.items() if feature_name in dnn_feature_names]),
        embedding_size=embedding_size
    )

    z_part = tf.concat(sparse_embeddings, axis=1)

    num_pairs = int(len(sparse_embeddings)*(len(sparse_embeddings)-1)/2)
    p_part = layers.PNN(mode=0, embedding_size=embedding_size, num_pairs=num_pairs)(sparse_embeddings)

    concat_embedding = tf.concat([z_part, p_part], axis=1)

    logits = common_layer.get_nn_layers(concat_embedding, dims=[128, 32, 1])

    logits = tf.squeeze(logits, axis=1)

    pred = tf.sigmoid(logits)

    predictions = {"prob": pred}

    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}

    """
        train & eval & predict
    """

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    else:
        # 定义loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_sum(loss)

        # 处理验证模式
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_hooks = [
                metrics.BinaryMetricHook(batch_size=batch_size, pred_tensor=pred, label_tensor=labels, prefix='Eval'),
            ]

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                evaluation_hooks=eval_hooks
            )

        # 处理训练模式
        if mode == tf.estimator.ModeKeys.TRAIN:

            training_hooks = [
                metrics.BinaryMetricHook(batch_size=batch_size, pred_tensor=pred, label_tensor=labels, prefix='Train'),
            ]

            optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                training_hooks=training_hooks
            )