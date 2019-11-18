import numpy as np
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear


def boxworld_cnn(scaled_images, **kwargs):
    """
    CNN for boxworld input[140,140].

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    print('scaled_images', scaled_images)
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=12, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=24, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    print('layer_3', layer_3)

    return layer_3


def rrl_cnn(scaled_images, **kwargs):
    """
    CNN from rrl paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    print('scaled_images', scaled_images)
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=12, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=24, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    print('layer_2', layer_2)
    return layer_2


def layerNorm(input_tensor, scope, eps=1e-5):
    """
    Creates a layerNormalization module for TensorFlow
    ref:https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/layer_norm.py

    :param input_tensor: (TensorFlow Tensor) The input tensor from NN [batch_size,layer_dim]
    :param scope: (str) The TensorFlow variable scope
    :param eps: (float) A small float number to avoid dividing by 0
    :return: (TensorFlow Tensor) layer Normalized optputs with same shape as input_tensor
    """
    with tf.variable_scope(scope):
        hidden_size = input_tensor.get_shape()[1].value
        gamma = tf.get_variable("gamma", [hidden_size], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", [hidden_size], initializer=tf.zeros_initializer())

        mean, var = tf.nn.moments(input_tensor, [1], keep_dims=True)
        normalized = tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, eps)
        return normalized


def get_coor(input_tensor):
    """
    The output of cnn is tagged with two extra channels indicating the spatial position(x and y) of each cell

    :param input_tensor: (TensorFlow Tensor) The input tensor from NN [B,Height,W,D]
    :return: (TensorFlow Tensor) [B,Height,W,2]
    """
    batch_size = tf.shape(input_tensor)[0]
    height = input_tensor.get_shape()[1].value
    width = input_tensor.get_shape()[2].value
    coor = []
    for h in range(height):
        w_channel = []
        for w in range(width):
            w_channel.append([float(h / height), float(w / width)])
        coor.append(w_channel)

    coor = tf.expand_dims(tf.constant(coor, dtype=input_tensor.dtype), axis=0)
    coor = tf.convert_to_tensor(coor)
    # [1,Height,W,2] --> [B,Height,W,2]
    coor = tf.tile(coor, [batch_size, 1, 1, 1])
    return coor


def MHDPA(input_tensor, scope, num_heads):
    """
    An implementation of the Multi-Head Dot-Product Attention architecture in "Relational Deep Reinforcement Learning"
    https://arxiv.org/abs/1806.01830
    ref to the RMC architecture on https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py

    :param input_tensor: (TensorFlow Tensor) The input tensor from NN [B,Height,W,D]
    :param scope: (str) The TensorFlow variable scope
    :param num_heads: (float) The number of attention heads to use
    :return: (TensorFlow Tensor) [B,Height*W,num_heads,D]
    """
    with tf.variable_scope(scope):
        last_num_height = input_tensor.get_shape()[1].value
        last_num_width = input_tensor.get_shape()[2].value
        last_num_features = input_tensor.get_shape()[3].value

        key_size = value_size = last_num_features
        qkv_size = 2 * key_size + value_size
        # total_size Denoted as F, num_heads Denoted as H
        total_size = qkv_size * num_heads

        # Denote N = last_num_height * last_num_width
        N = last_num_height * last_num_width
        # [B*N,Deepth]
        extracted_features_reshape = tf.reshape(input_tensor, [-1, last_num_features])
        # [B*N,F]
        qkv = linear(extracted_features_reshape, "QKV", total_size)
        # [B*N,F]
        qkv = layerNorm(qkv, "qkv_layerNorm")
        # [B,N,F]
        qkv = tf.reshape(qkv, [-1, last_num_height * last_num_width, total_size])
        # [B,N,H,F/H]
        qkv_reshape = tf.reshape(qkv, [-1, last_num_height * last_num_width, num_heads, qkv_size])
        print("qkv_reshape", qkv_reshape)
        # [B,N,H,F/H] -> [B,H,N,F/H]
        qkv_transpose = tf.transpose(qkv_reshape, [0, 2, 1, 3])
        print("qkv_transpose", qkv_transpose)
        q, k, v = tf.split(qkv_transpose, [key_size, key_size, value_size], -1)

        # q *= qkv_size ** -0.5
        # [B,H,N,N]
        dot_product = tf.matmul(q, k, transpose_b=True)
        dot_product = dot_product * (N**-0.5)
        weights = tf.nn.softmax(dot_product)
        # [B,H,N,V]
        output = tf.matmul(weights, v)
        # [B,H,N,V] -> [B,N,H,V]
        output_transpose = tf.transpose(output, [0, 2, 1, 3])

        return output_transpose, weights


def residual_block(x, y):
    """
    Z = W*y + x
    :param x: (TensorFlow Tensor) The input tensor from NN [B,Height,W,D]
    :param y: (TensorFlow Tensor) The input tensor from MHDPA [B,Height*W,num_heads,D]
    :return: (TensorFlow Tensor) [B,Height*W,num_heads,D]
    """
    last_num_height = x.get_shape()[1].value
    last_num_width = x.get_shape()[2].value
    last_num_features = x.get_shape()[3].value
    # W*y
    y_Matmul_W = conv(y, 'y_Matmul_W', n_filters=last_num_features, filter_size=1, stride=1, init_scale=np.sqrt(2))
    print('y_Matmul_W', y_Matmul_W)
    # [B,Height,W,D] --> [B,Height*W,D]
    x_reshape = tf.reshape(x, [-1, last_num_width * last_num_height, last_num_features])
    x_edims = tf.expand_dims(x_reshape, axis=2)
    num_heads = y.get_shape()[2]
    # [B,Height*W,1,D] --> [B,Height*W,H,D]
    x_edims = tf.tile(x_edims, [1,  1, num_heads, 1])
    # W*y + x
    residual_output = tf.add(y_Matmul_W, x_edims)
    return residual_output
