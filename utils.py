import numpy as np
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear


def concise_cnn(scaled_images, **kwargs):
    """
    concise CNN, output = input = [W,H,C].

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    print('scaled_images', scaled_images)
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
    print('layer_2', layer_2)
    return layer_2


def boxworld_cnn(scaled_images, **kwargs):
    """
    CNN for boxworld input(scaled_images): [140,140] .

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    print('scaled_images', scaled_images)
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    print('layer_3', layer_3)

    return layer_3


def simple_cnn(scaled_images, **kwargs):
    """
    simple CNN, input = [14*4,14*4,C].

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    print('scaled_images', scaled_images)
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=12, filter_size=2, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=24, filter_size=2, stride=2, init_scale=np.sqrt(2), **kwargs))
    print('layer_2', layer_2)
    return layer_2


def concise_cnn(scaled_images, **kwargs):
    """
    concise CNN, input = [14,14,C].

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
    return layer_2


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
        hidden_size = input_tensor.shape[1].value
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
    _, height, width, _ = input_tensor.shape.as_list()
    coor = [[[h / height, w / width] for w in range(width)] for h in range(height)]
    # coor = []
    # for h in range(height):
    #     w_channel = []
    #     for w in range(width):
    #         w_channel.append([float(h / height), float(w / width)])
    #     coor.append(w_channel)
    coor = tf.expand_dims(tf.constant(coor, dtype=input_tensor.dtype), axis=0)
    coor = tf.convert_to_tensor(coor)
    # [1,Height,W,2] --> [B,Height,W,2]
    coor = tf.tile(coor, [batch_size, 1, 1, 1])
    return coor


def MHDPA(entities, scope, num_heads):
    """
    same as MHDPA, the difference is input tensor:[B,N,D]
    :param entities: (TensorFlow Tensor) The input tensor from NN [B,N,D]
    :param scope: (str) The TensorFlow variable scope
    :param num_heads: (float) The number of attention heads to use
    :return: (TensorFlow Tensor) [B,N,num_heads,D]
    """
    with tf.variable_scope(scope):
        N = entities.shape[1].value
        query_size = key_size = value_size = channels = entities.shape[2].value
        qkv_size = query_size + key_size + value_size
        # total_size Denoted as F, num_heads Denoted as H
        total_size = qkv_size * num_heads

        # [B*N,Deepth]
        entities = tf.reshape(entities, [-1, channels])
        # [B*N,F] F = 3*D*n_Head
        qkv = linear(entities, "QKV", total_size)
        # [B*N,F]
        qkv = layerNorm(qkv, "qkv_layerNorm")
        # # [B,N,F]
        # qkv = tf.reshape(qkv, [-1, N, total_size])
        # [B,N,H,F/H]
        qkv = tf.reshape(qkv, [-1, N, num_heads, qkv_size])
        # [B,N,H,F/H] -> [B,H,N,F/H]
        qkv = tf.transpose(qkv, [0, 2, 1, 3])
        # q = k = v = [B,H,N,D]
        q, k, v = tf.split(qkv, [key_size, key_size, value_size], -1)

        # q *= qkv_size ** -0.5
        # [B,H,N,N]
        dot_product = tf.matmul(q, k, transpose_b=True)
        dot_product = dot_product * (query_size**-0.5)
        weights = tf.nn.softmax(dot_product)
        # [B,H,N,V]
        output = tf.matmul(weights, v)
        # [B,H,N,V] -> [B,N,H,V]
        output = tf.transpose(output, [0, 2, 1, 3])

        return output, weights


def residual_block(x, y):
    """
    Z = W*y + x
    :param x: (TensorFlow Tensor) The input tensor from CNN [B,N,D] N = n_entities
    :param y: (TensorFlow Tensor) The input tensor from MHDPA [B,N,num_heads,D]
    :return: (TensorFlow Tensor) [B,N,num_heads,D]
    """
    # W*y
    y_Matmul_W = conv(y, 'y_Matmul_W', n_filters=y.shape[3].value, filter_size=1, stride=1, init_scale=np.sqrt(2))
    # print('y_Matmul_W', y_Matmul_W)
    x_edims = tf.expand_dims(x, axis=2)
    # [B,N,1,D] --> [B,N,H,D]
    x_edims = tf.tile(x_edims, [1,  1, y.shape[2].value, 1])
    # W*y + x
    residual_output = tf.add(y_Matmul_W, x_edims)
    return residual_output


def reduce_border_extractor(input_tensor):
    """
    reduce boxworld border, and concat indcator
    """
    # --- inter ---
    gs = input_tensor.shape[1].value // 14
    # [B,14*gs,14*gs,3] --> [B,12*gs,12*gs,3]
    inter = input_tensor[:, gs:-gs, gs:-gs, :]
    # [B,W,H,D]
    inter = concise_cnn(inter)
    # [B,W*H,D]
    inter_entities = entities_flatten(inter)
    # --- indicator ---
    # [B,3]
    indicator = input_tensor[:, 0, 0, :]
    # [B,D]
    indicator = linear(indicator, "indicator", inter.shape[3].value)
    # [B,D] -->[B,1,D]
    indicator_entity = tf.expand_dims(indicator, axis=1)
    # [B,W*H+1,D]
    entities = tf.concat([inter_entities, indicator_entity], axis=1)
    return entities


def build_entities(processed_obs, reduce_obs=False):
    coor = get_coor(processed_obs)
    if reduce_obs:
        # [B,Height,W,D+2]
        processed_obs = tf.concat([processed_obs, coor], axis=3)
        # [B,N,D] N=Height*w+1
        entities = reduce_border_extractor(processed_obs)
    else:
        # [B,Height,W,D]
        extracted_features = concise_cnn(processed_obs)
        # [B,Height,W,D+2]
        entities = tf.concat([extracted_features, coor], axis=3)
        # [B,N,D] N=Height*w
        entities = entities_flatten(entities)
    return entities


def entities_flatten(input_tensor):
    """
    flatten axis 1 and axis 2
    :param input_tensor: (TensorFlow Tensor) The input tensor from NN [B,H,W,D]
    :return: (TensorFlow Tensor) [B,N,D]
    """
    _, h, w, channels = input_tensor.shape.as_list()
    return tf.reshape(input_tensor, [-1, h * w, channels])
