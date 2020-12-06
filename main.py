# coding=utf-8
"""
-------------------------------------
author : yezhiwen.buaa
introduction : 
-------------------------------------
"""

import tensorflow as tf
from util import input_util, model_util
from model import MLP

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("model_dir", 'save_model/MLP', 'model_dir')
tf.app.flags.DEFINE_integer("train", 1, 'train mode')
tf.app.flags.DEFINE_integer("eval", 1, 'eval mode')
tf.app.flags.DEFINE_integer("epoch", 1, 'epoch')
tf.app.flags.DEFINE_integer("batch_size", 256, 'batch_size')


def main(_):
    input_fn = input_util.input_fn
    model_fn = MLP.model_fn

    # 处理输入
    path = "data/criteo_sample.txt"
    sparse_features, dense_features, sparse_feature_columns, dense_feature_columns, label = input_util.load_criteo_data(path, sep='\t')

    # 划分训练集 & 验证集
    train_size = int(len(label) * 0.85)
    train_sparse_features = sparse_features[:train_size]
    train_dense_features = dense_features[:train_size]
    train_label = label[:train_size]
    eval_sparse_features = sparse_features[train_size:]
    eval_dense_features = dense_features[train_size:]
    eval_label = label[train_size:]

    # 处理model
    model_params = {
        "sparse_feature_columns": sparse_feature_columns,
        "dense_feature_columns": dense_feature_columns,
        "batch_size": FLAGS.batch_size,
        "lr": 0.01
    }

    # 清空model dir
    model_util.clean_model(FLAGS.model_dir)

    config = tf.estimator.RunConfig()
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.train == 1:
        estimator.train(
            input_fn=lambda: input_fn(train_sparse_features, train_dense_features, train_label, epoch=FLAGS.epoch,
                                      batch_size=FLAGS.batch_size)
        )

    if FLAGS.eval == 1:
        estimator.evaluate(
            input_fn=lambda: input_fn(eval_sparse_features, eval_dense_features, eval_label, epoch=FLAGS.epoch,
                                      batch_size=FLAGS.batch_size)
        )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)