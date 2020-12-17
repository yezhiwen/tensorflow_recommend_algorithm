# coding=utf-8
"""
-------------------------------------
author : yezhiwen.buaa
introduction : 
-------------------------------------
"""
import tensorflow as tf
from tensorflow.python.keras import backend as K

class Example(tf.keras.layers.Layer):

    def __init__(self, **kwargs):

        super(Example, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        pass

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

class MMoE(tf.keras.layers.Layer):

    def __init__(self, units, num_experts, num_tasks,
                 use_expert_bias=True, use_gate_bias=True, expert_activation='relu', gate_activation='softmax',
                 expert_bias_initializer='zeros', gate_bias_initializer='zeros', expert_bias_regularizer=None,
                 gate_bias_regularizer=None, expert_bias_constraint=None, gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling', gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None, gate_kernel_regularizer=None, expert_kernel_constraint=None,
                 gate_kernel_constraint=None, activity_regularizer=None, **kwargs):
        super(MMoE, self).__init__(**kwargs)

        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = tf.keras.initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = tf.keras.initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = tf.keras.regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = tf.keras.regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = tf.keras.constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = tf.keras.constraints.get(gate_kernel_constraint)

        # Activation parameter
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = tf.keras.initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = tf.keras.initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = tf.keras.regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = tf.keras.regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = tf.keras.constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = tf.keras.constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []

        for i in range(self.num_experts):
            self.expert_layers.append(tf.keras.layers.Dense(self.units, activation=self.expert_activation,
                                                   use_bias=self.use_expert_bias,
                                                   kernel_initializer=self.expert_kernel_initializer,
                                                   bias_initializer=self.expert_bias_initializer,
                                                   kernel_regularizer=self.expert_kernel_regularizer,
                                                   bias_regularizer=self.expert_bias_regularizer,
                                                   activity_regularizer=None,
                                                   kernel_constraint=self.expert_kernel_constraint,
                                                   bias_constraint=self.expert_bias_constraint))
        for i in range(self.num_tasks):
            self.gate_layers.append(tf.keras.layers.Dense(self.num_experts, activation=self.gate_activation,
                                                 use_bias=self.use_gate_bias,
                                                 kernel_initializer=self.gate_kernel_initializer,
                                                 bias_initializer=self.gate_bias_initializer,
                                                 kernel_regularizer=self.gate_kernel_regularizer,
                                                 bias_regularizer=self.gate_bias_regularizer, activity_regularizer=None,
                                                 kernel_constraint=self.gate_kernel_constraint,
                                                 bias_constraint=self.gate_bias_constraint))

    def call(self, inputs, **kwargs):

        expert_outputs, gate_outputs, final_outputs = [], [], []

        # inputs: (batch_size, embedding_size)
        for expert_layer in self.expert_layers:
            expert_output = tf.expand_dims(expert_layer(inputs), axis=2)
            expert_outputs.append(expert_output)

        # batch_size * units * num_experts
        expert_outputs = tf.concat(expert_outputs, 2)

        # [(batch_size, num_experts), ......]
        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(inputs))

        for gate_output in gate_outputs:
            # (batch_size, 1, num_experts)
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)

            # (batch_size * units * num_experts) * (batch_size, 1 * units, num_experts)
            weighted_expert_output = expert_outputs * tf.keras.backend.repeat_elements(expanded_gate_output, self.units, axis=1)

            # (batch_size, units)
            final_outputs.append(tf.reduce_sum(weighted_expert_output, axis=2))

        # [(batch_size, units), ......]   size: num_task
        return final_outputs

class AFM(tf.keras.layers.Layer):

    def __init__(self, embedding_size, attention_size, l2_reg=0.0, **kwargs):

        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.l2_reg = l2_reg

        super(AFM, self).__init__(**kwargs)

    def build(self, input_shape):

        self.attention_w = self.add_weight(
                'attention_w',
                shape=[self.embedding_size, self.attention_size],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True)

        self.attention_b = self.add_weight(
            'attention_b',
            shape=[self.attention_size],
            initializer=tf.keras.initializers.glorot_normal(),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
            trainable=True)

        self.attention_h = self.add_weight(
            'attention_h',
            shape=[self.attention_size],
            initializer=tf.keras.initializers.glorot_normal(),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
            trainable=True)

        self.attention_p = self.add_weight(
            'attention_p',
            shape=[self.embedding_size],
            initializer=tf.keras.initializers.glorot_normal(),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
            trainable=True)

    def call(self, inputs, **kwargs):

        # [(batch_size, embedding_size), ......]   size: feature_size
        embedding_list = inputs

        cross_embeddings = []
        for i in range(len(embedding_list)-1):
            for j in range(len(embedding_list)):
                cross_embedding = tf.multiply(embedding_list[i], embedding_list[j])
                cross_embeddings.append(cross_embedding)

        # (feature_size(feature_size - 1)/2, batch_size, embedding_size)
        cross_embeddings = tf.stack(cross_embeddings)

        # (batch_size, feature_size(feature_size - 1)/2, embedding_size)
        cross_embeddings = tf.transpose(cross_embeddings, perm=[1, 0, 2])

        # (batch_size, feature_size(feature_size - 1)/2, attention_size)
        attention_weight = tf.add(tf.matmul(cross_embeddings, self.attention_w), self.attention_b)

        # (batch_size, feature_size(feature_size - 1)/2)
        attention_a = tf.nn.softmax(tf.reduce_sum(tf.multiply(tf.nn.relu(attention_weight), self.attention_h), axis=2))

        # (batch_size, feature_size(feature_size - 1)/2, 1)
        attention_out = tf.expand_dims(attention_a, axis=2)

        # (batch_size, embedding_size)
        attention_out = tf.reduce_sum(tf.multiply(cross_embeddings, attention_out), axis=1)

        # (batch_size, 1)
        attention_out = tf.reduce_sum(tf.multiply(attention_out, self.attention_p), axis=1, keepdims=True)

        return attention_out

class PNN(tf.keras.layers.Layer):

    def __init__(self, mode, embedding_size, num_pairs, num_embeddings, **kwargs):

        # 指定是 IPNN or OPNN
        self.mode = mode

        self.num_embeddings = num_embeddings
        self.num_pairs = num_pairs
        self.embedding_size = embedding_size

        super(PNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.opnn_weight = self.add_weight(
                'opnn_weight',
                shape=[self.embedding_size, self.num_pairs, self.embedding_size],
                initializer=tf.keras.initializers.glorot_normal(),
                trainable=True
        )

    def call(self, inputs, **kwargs):

        embeddings = inputs

        if self.mode == 0:
            """
                IPNN
            """
            # embedding: (batch_size, embedding_size) -> (batch_size, 1, embedding_size)
            embeddings = [tf.expand_dims(embedding, axis=1) for embedding in embeddings]

            row = []
            col = []
            num_inputs = len(embeddings)

            for i in range(self.num_embeddings - 1):
                for j in range(i + 1, self.num_embeddings):
                    row.append(i)
                    col.append(j)

            # batch_size num_pairs embedding_size
            p = tf.concat([embeddings[idx] for idx in row], axis=1)
            q = tf.concat([embeddings[idx] for idx in col], axis=1)

            # batch_size num_pairs embedding_size
            inner_product = p * q

            # batch_size num_pairs
            inner_product = tf.reduce_sum(inner_product, axis=-1)

            return inner_product

        elif self.mode == 1:
            """
                OPNN
            """

            # embedding: (batch_size, embedding_size) -> (batch_size, 1, embedding_size)
            embeddings = [tf.expand_dims(embedding, axis=1) for embedding in embeddings]

            # batch_size, feature_size, embedding_size
            embeddings = tf.concat(embeddings, axis=1)

            row = []
            col = []

            for i in range(self.num_embeddings - 1):
                for j in range(i + 1, self.num_embeddings):
                    row.append(i)
                    col.append(j)

            # batch_size, num_pairs, embedding_size
            p = tf.transpose(
                # num_pairs, batch_size, embedding_size
                tf.gather(
                    # feature_size, batch_size, embedding_size
                    tf.transpose(embeddings, [1, 0, 2]),
                    row),
                [1, 0, 2])

            # batch_size, num_pairs, embedding_size
            q = tf.transpose(
                tf.gather(
                    tf.transpose(embeddings, [1, 0, 2]),
                    col),
                [1, 0, 2])

            # batch_size * 1 * num_pairs * embedding_size
            p = tf.expand_dims(p, axis=1)

            # batch_size * num_pairs
            outer_product = tf.reduce_sum(
                # batch_size * num_pairs * embedding_size
                tf.multiply(
                    # batch_size * num_pairs * embedding_size
                    tf.transpose(
                        # batch_size * embedding_size * num_pairs
                        tf.reduce_sum(
                            # (batch_size * 1 * num_pairs * embedding_size)
                            # * (embedding_size * num_pairs * embedding_size)
                            # -> batch_size * embedding_size * num_pairs * embedding_size
                            tf.multiply(p, self.opnn_weight),
                            axis=-1),
                        [0, 2, 1]),
                    q),
                axis=-1)

            return outer_product

        else:
            raise Exception("PNN mode is not in [0, 1]")



