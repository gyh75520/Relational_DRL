import numpy as np
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear


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
    input = [H,W,D] H=W
    output = [H,W,64]
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
    return layer_2


def deepconcise_cnn(scaled_images, **kwargs):
    """
    input = [H,W,D] H=W
    output = [H-3,W-3,64]
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
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


# def layerNorm(input_tensor, scope, eps=1e-5):
#     """
#     :param input_tensor: (TensorFlow Tensor) The input tensor  [batch_size,layer_dim]
#     :param scope: (str) The TensorFlow variable scope
#     :param eps: (float) A small float number to avoid dividing by 0
#     :return: (TensorFlow Tensor) layer Normalized optputs with same shape as input_tensor
#     """
#     with tf.variable_scope(scope):
#         hidden_size = input_tensor.shape[-1].value
#         gamma = tf.get_variable("gamma", [hidden_size], initializer=tf.ones_initializer())
#         beta = tf.get_variable("beta", [hidden_size], initializer=tf.zeros_initializer())
#
#         mean, var = tf.nn.moments(input_tensor, [1], keep_dims=True)
#         normalized = tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, eps)
#         return normalized

def layerNorm(input_tensor, scope, eps=1e-5):
    """
    gamma, beta = [D]
    mean, var's axis = [2]
    :param input_tensor: (TensorFlow Tensor) The input tensor [B,N,D]]
    :param scope: (str) The TensorFlow variable scope
    :param eps: (float) A small float number to avoid dividing by 0
    :return: (TensorFlow Tensor) layer Normalized optputs with same shape as input_tensor
    """
    with tf.variable_scope(scope):
        hidden_size = input_tensor.shape.as_list()[-1:]
        gamma = tf.get_variable("gamma", hidden_size, initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", hidden_size, initializer=tf.zeros_initializer())

        mean, var = tf.nn.moments(input_tensor, [2], keep_dims=True)
        print('layerNorm_mean', mean.shape)
        normalized = tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, eps)
        return normalized


def layerNorm2(input_tensor, scope, eps=1e-5):
    """
    gamma, beta = [N,D]
    mean, var's axis = [1,2]
    :param input_tensor: (TensorFlow Tensor) The input tensor [B,N,D]]
    :param scope: (str) The TensorFlow variable scope
    :param eps: (float) A small float number to avoid dividing by 0
    :return: (TensorFlow Tensor) layer Normalized optputs with same shape as input_tensor
    """
    with tf.variable_scope(scope):
        hidden_size = input_tensor.shape.as_list()[-2:]
        gamma = tf.get_variable("gamma", hidden_size, initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", hidden_size, initializer=tf.zeros_initializer())

        mean, var = tf.nn.moments(input_tensor, [1, 2], keep_dims=True)
        print('layerNorm2_mean', mean.shape)
        normalized = tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, eps)
        return normalized


def layerNorm3(input_tensor, scope, eps=1e-5):
    """
    gamma, beta = [D]
    mean, var's axis = [1,2]
    :param input_tensor: (TensorFlow Tensor) The input tensor [B,N,D]]
    :param scope: (str) The TensorFlow variable scope
    :param eps: (float) A small float number to avoid dividing by 0
    :return: (TensorFlow Tensor) layer Normalized optputs with same shape as input_tensor
    """
    with tf.variable_scope(scope):
        hidden_size = input_tensor.shape.as_list()[-1:]
        gamma = tf.get_variable("gamma", hidden_size, initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", hidden_size, initializer=tf.zeros_initializer())

        mean, var = tf.nn.moments(input_tensor, [1, 2], keep_dims=True)
        print('layerNorm3_mean', mean.shape)
        normalized = tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, eps)
        return normalized


def batchNorm(input_tensor, scope, eps=1e-5):
    """
    gamma, beta = [D]
    mean, var's axis = [0,1]
    :param input_tensor: (TensorFlow Tensor) The input tensor [B,N,D]
    :param scope: (str) The TensorFlow variable scope
    :param eps: (float) A small float number to avoid dividing by 0
    :return: (TensorFlow Tensor) layer Normalized optputs with same shape as input_tensor
    """
    with tf.variable_scope(scope):
        hidden_size = input_tensor.shape.as_list()[-1:]
        gamma = tf.get_variable("gamma", hidden_size, initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", hidden_size, initializer=tf.zeros_initializer())

        mean, var = tf.nn.moments(input_tensor, [0, 1], keep_dims=True)
        print('batchNorm_mean', mean.shape)
        normalized = tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, eps)
        return normalized


def instanceNorm(input_tensor, scope, eps=1e-5):
    """
    gamma, beta = [D]
    mean, var's axis = [1]
    :param input_tensor: (TensorFlow Tensor) The input tensor [B,N,D]
    :param scope: (str) The TensorFlow variable scope
    :param eps: (float) A small float number to avoid dividing by 0
    :return: (TensorFlow Tensor) layer Normalized optputs with same shape as input_tensor
    """
    with tf.variable_scope(scope):
        hidden_size = input_tensor.shape.as_list()[-1:]
        gamma = tf.get_variable("gamma", hidden_size, initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", hidden_size, initializer=tf.zeros_initializer())

        mean, var = tf.nn.moments(input_tensor, [1], keep_dims=True)
        print('instanceNorm_mean', mean.shape)
        normalized = tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, eps)
        return normalized


def get_coor(input_tensor):
    """
    The output of cnn is tagged with two extra channels indicating the spatial position(x and y) of each cell

    :param input_tensor: (TensorFlow Tensor)  [B,Height,W,D]
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


def embedding(entities, n_heads, embedding_sizes, scope):
    """
    :param entities: (TensorFlow Tensor) The input entities : [B,N,D]
    :param scope: (str) The TensorFlow variable scope
    :param n_heads: (float) The number of attention heads to use
    :return: (TensorFlow Tensor) [B,n_heads,N,embedding_sizes[i]]
    """
    with tf.variable_scope(scope):
        N = entities.shape[1].value
        channels = entities.shape[2].value
        # total_size Denoted as F, n_heads Denoted as H
        total_size = sum(embedding_sizes) * n_heads
        # [B*N,D]
        entities = tf.reshape(entities, [-1, channels])
        # [B*N,F] F = sum(embedding_sizes) * n_heads
        embedded_entities = linear(entities, "mlp", total_size)
        # [B*N,F] --> [B,N,F] new
        embedded_entities = tf.reshape(embedded_entities, [-1, N, total_size])
        # [B*N,F]
        qkv = layerNorm(embedded_entities, "ln")
        # qkv = batchNorm(embedded_entities, "bn")
        # qkv = instanceNorm(embedded_entities, "instacne_n")
        # # [B,N,F]
        # qkv = tf.reshape(qkv, [-1, N, total_size])
        # [B,N,n_heads,sum(embedding_sizes)]
        qkv = tf.reshape(qkv, [-1, N, n_heads, sum(embedding_sizes)])
        # [B,N,n_heads,sum(embedding_sizes)] -> [B,n_heads,N,sum(embedding_sizes)]
        qkv = tf.transpose(qkv, [0, 2, 1, 3])
        return tf.split(qkv, embedding_sizes, -1)


def getQKV(entities, n_heads, scope):
    """
    :param entities: (TensorFlow Tensor) The input entities : [B,N,D]
    :param scope: (str) The TensorFlow variable scope
    :param n_heads: (float) The number of attention heads to use
    :return: (TensorFlow Tensor) [B,n_heads,N,D]
    """
    query_size = key_size = value_size = entities.shape[-1].value
    return embedding(entities, n_heads, [query_size, key_size, value_size], scope)


def updateRelations(dot_product, v):
    """
    :param entities: (TensorFlow Tensor) dot_product: [B,n_heads,N1,N2]
    :return: (TensorFlow Tensor) [B,n_heads,N,D]
    """
    # [B,n_heads,N1,N2]
    relations = tf.nn.softmax(dot_product)
    # [B,n_heads,N1,D]
    output = tf.matmul(relations, v)
    # # [B,n_heads,N1,D] -> [B,n_heads,N1,D]
    # output = tf.transpose(output, [0, 2, 1, 3])
    return output, relations


def MHDPA(entities,  n_heads):
    """
    An implementation of the Multi-Head Dot-Product Attention architecture in "Relational Deep Reinforcement Learning"
    https://arxiv.org/abs/1806.01830
    ref to the RMC architecture on https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py

    :param entities: (TensorFlow Tensor) entities [B,N,D]
    :param n_heads: (float) The number of attention heads to use
    :return: (TensorFlow Tensor) [B,n_heads,N,D]
    """
    q, k, v = getQKV(entities, n_heads, 'QKV')
    # dot_product *= qkv_size ** -0.5
    # [B,n_heads,N,N]
    dot_product = tf.matmul(q, k, transpose_b=True)
    channels = v.shape[-1].value
    dot_product = dot_product * (channels**-0.5)
    return updateRelations(dot_product, v)


def residualNet(x, y, scope):
    """
    Z = W*y + x
    :param x: (TensorFlow Tensor) entities [B,n_heads,N,D]
    :param y: (TensorFlow Tensor) new_entities from MHDPA [B,n_heads,N,D]
    :return: (TensorFlow Tensor) [B,n_heads,N,D] or [B,N,n_heads,D]
    """
    with tf.variable_scope(scope):
        # W*y
        output = conv(y, 'y_Matmul_W', n_filters=y.shape[3].value, filter_size=1, stride=1, init_scale=np.sqrt(2))
        # W*y + x
        output = tf.add(output, x)
        return output


def residual_block(x, y):
    """
    Z = W*y + x
    :param x: (TensorFlow Tensor) entities [B,N,D] N = n_entities
    :param y: (TensorFlow Tensor) new_entities from MHDPA [B,n_heads,N,D]
    :return: (TensorFlow Tensor) [B,n_heads,N,D]
    """
    x_edims = tf.expand_dims(x, axis=1)
    # [B,1,N,D] --> [B,n_heads,N,D]
    x_edims = tf.tile(x_edims, [1,  y.shape[1].value, 1, 1])
    return residualNet(x_edims, y, 'residualNet')


def reduce_border_extractor(input_tensor, cnn_extractor):
    """
    reduce boxworld border, and concat indcator
    """
    # --- inter ---
    gs = input_tensor.shape[1].value // 14
    # [B,14*gs,14*gs,3] --> [B,12*gs,12*gs,3]
    inter = input_tensor[:, gs:-gs, gs:-gs, :]
    # [B,W,H,D]
    inter = cnn_extractor(inter)
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
    cnn_extractor = deepconcise_cnn
    if reduce_obs:
        # [B,Height,W,D+2]
        processed_obs = tf.concat([processed_obs, coor], axis=3)
        # [B,N,D] N=Height*w+1
        entities = reduce_border_extractor(processed_obs, cnn_extractor)
    else:
        # [B,Height,W,D]
        extracted_features = cnn_extractor(processed_obs)
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
