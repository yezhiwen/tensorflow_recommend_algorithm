# coding=utf-8
"""
-------------------------------------
author : yezhiwen.buaa
introduction : 
-------------------------------------
"""
import tensorflow as tf
from tensorflow.python.keras import backend as K

class Linear(tf.keras.layers.Layer):

    def __init__(self, l2_reg=0.0, mode=0, use_bias=False, **kwargs):

        self.l2_reg = l2_reg

        if mode not in [0, 1, 2]:
            raise ValueError("mode must be 0,1 or 2")

        self.mode = mode
        self.use_bias = use_bias
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        if self.mode == 1:
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=[int(input_shape[-1]), 1],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True)
        elif self.mode == 2 :
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=[int(input_shape[1][-1]), 1],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True)

        super(Linear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.mode == 0:
            sparse_input = inputs
            linear_logit = tf.reduce_sum(sparse_input, axis=-1, keep_dims=True)
        elif self.mode == 1:
            dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(1, 0))
            linear_logit = fc
        else:
            sparse_input, dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(1, 0))
            linear_logit = tf.reduce_sum(sparse_input, axis=-1, keep_dims=True) + fc
        if self.use_bias:
            linear_logit += self.bias

        return linear_logit


class FM(tf.keras.layers.Layer):

    """
        input: (None, feature_size, embedding_size)
        output: (None, 1)
    """

    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        # None * 1 * embedding_size
        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))

        # None * 1 * embedding_size
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)

        cross_term = square_of_sum - sum_of_square

        # None * 1
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term