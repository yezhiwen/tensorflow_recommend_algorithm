# coding=utf-8
"""
-------------------------------------
author : yezhiwen.buaa
introduction : FM
-------------------------------------
"""


import tensorflow as tf
from layer import layers
from metrics import metrics
from util import model_util


def model_fn(features, labels, mode, params):

    """
    :param features: input_fn 得到的feature
    :param labels: labels
    :param mode: 当前模式：分为训练、验证、预测
    :param params: 模型的其他参数
    :return: estimator
    """

    """
        输入feature & 配置的读取
    """

    # 获取 feature tensor
    # 1. 对于 sparse_feature 存储的是index
    # 2. 对于 dense_feature 存储的是具体的值
    sparse_features_input = features['sparse_features']

    # 获取基本配置
    batch_size = params['batch_size']
    lr = params['lr']
    embedding_size = params['embedding_size']
    attention_size = params['attention_size']

    # 获取 sparse feature 配置
    sparse_feature_columns = params['sparse_feature_columns']

    # 获取线性部分logits
    linear_logits = model_util.get_linear_logits(
        sparse_feature_columns=sparse_feature_columns,
        sparse_features_input=dict([(feature_name, input) for feature_name, input in sparse_features_input.items()]),
        dense_features_input=dict()
    )

    # 创建 sparse feature embedding
    sparse_embeddings = model_util.get_feature_embeddings(
        sparse_feature_columns,
        dict([(feature_name, input) for feature_name, input in sparse_features_input.items()]),
        embedding_size=embedding_size
    )

    # 获取 AFM logits

    afm_logits = layers.AFM(embedding_size=embedding_size, attention_size=attention_size)(sparse_embeddings)

    logits = tf.add_n([linear_logits, afm_logits])

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