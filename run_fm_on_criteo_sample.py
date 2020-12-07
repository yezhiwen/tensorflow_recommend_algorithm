# coding=utf-8
"""
-------------------------------------
author : yezhiwen.buaa
introduction : 
-------------------------------------
"""

import tensorflow as tf
from util import input_util, model_util
from model.fm import model_fn
from util.input_util import input_fn

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("model_dir", 'save_model/FM', 'model_dir')
tf.app.flags.DEFINE_integer("train", 1, 'train mode')
tf.app.flags.DEFINE_integer("eval", 1, 'eval mode')
tf.app.flags.DEFINE_integer("epoch", 1, 'epoch')
tf.app.flags.DEFINE_integer("batch_size", 256, 'batch_size')


def main(_):

    """
        数据输入部分
    """
    # 数据集路径
    path = "data/criteo_sample.txt"

    # 加载数据集
    sparse_features, dense_features, sparse_feature_columns, dense_feature_columns, label = input_util.load_criteo_data(path, sep='\t')

    # 划分训练集 & 验证集
    train_size = int(len(label) * 0.85)
    train_sparse_features = sparse_features[:train_size]
    train_dense_features = dense_features[:train_size]
    train_label = label[:train_size]
    eval_sparse_features = sparse_features[train_size:]
    eval_dense_features = dense_features[train_size:]
    eval_label = label[train_size:]

    # 指定 wide & deep 模型的 wide部分和deep部分所需要的feature
    linear_feature_names = [each.name for each in sparse_feature_columns] + [each.name for each in dense_feature_columns]
    dnn_feature_names =  [each.name for each in sparse_feature_columns]

    # 处理model params
    model_params = {
        "sparse_feature_columns": sparse_feature_columns,
        "batch_size": FLAGS.batch_size,
        "embedding_size": 8,
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