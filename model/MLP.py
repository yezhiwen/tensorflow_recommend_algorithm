# coding=utf-8
"""
-------------------------------------
author : yezhiwen.buaa
introduction : 
-------------------------------------
"""

from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import RandomNormal
import tensorflow as tf
from layer import common_layer
from metrics import metrics

def creative_sparse_embedding_dict(sparse_feature_columns, l2_reg_embedding=0.00001, init_std=0.0001, seed=1024):
    sparse_embedding_dict = {feat.embedding_name:
                             Embedding(
                                 feat.vocabulary_size,
                                 feat.embedding_dim,
                                 embeddings_initializer=RandomNormal( mean=0.0, stddev=init_std, seed=seed),
                                 embeddings_regularizer=l2(l2_reg_embedding),
                                 name='emb_' + feat.embedding_name
                             )
                        for feat in sparse_feature_columns}
    return sparse_embedding_dict


def sparse_embedding_lookup(sparse_embedding_dict, sparse_features):
    embeddings = []
    for feature_name in sparse_features:
        lookup_idx = sparse_features[feature_name]
        embeddings.append(sparse_embedding_dict[feature_name](lookup_idx))
    return embeddings


def model_fn(features, labels, mode, params):

    # 获取基本配置
    batch_size = params['batch_size']
    lr = params['lr']


    # 获取 feature 配置
    sparse_feature_columns = params['sparse_feature_columns']
    dense_feature_columns = params['dense_feature_columns']

    # 创建 sparse feature embedding
    sparse_embedding_dict = creative_sparse_embedding_dict(sparse_feature_columns)

    # 获取 feature tensor
    sparse_features = features['sparse_features']
    dense_features = features['dense_features']
    dense_features = [tf.expand_dims(value, axis=1) for key,value in dense_features.items()]

    # 获取embeddings
    sparse_embeddings = sparse_embedding_lookup(sparse_embedding_dict, sparse_features)
    sparse_embeddings.extend(dense_features)

    concat_embeddings = tf.concat(sparse_embeddings, axis=1)
    logits = common_layer.get_nn_layers(concat_embeddings, dims=[1])
    print("logits", logits)

    logits = tf.squeeze(logits, axis=1)

    pred = tf.sigmoid(logits)

    predictions = {"prob": pred}

    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}

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