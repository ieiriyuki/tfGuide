#!/usr/local/bin/python

import os
import tensorflow as tf

def batch(batch_size=32):
    paths = []
    # ここではラベル番号は不要なので、ファイルパスだけ収集
    topdir = os.path.join('images_background_small2', 'Japanese_(katakana)')
    for dirpath, _, files in os.walk(topdir, followlinks=True):
        paths += [os.path.join(dirpath, file) for file in files]
    # 32x32に縮小して画像のみのバッチを生成
    queue = tf.train.slice_input_producer([paths])
    png = tf.read_file(queue[0])
    image = tf.image.decode_png(png, channels=1)
    image = tf.image.resize_images(image, [32, 32])
    image = tf.substract(tf.divide(image, 127.5), 1.0)
    return tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        capacity=len(paths) + 3 * batch_size,
        min_after_dequeue=len(paths)
    )

def generator(inputs, batch_size):
    '''
    乱数ベクトルから画像を生成するモデル
    Args:
        inputs: [batch_size, 10]のテンソル
        batch_size: 32
    Returns:
        生成結果の[batch_size, 32, 32, 1]のテンソル
    '''
    with tf.variable_scope('g'):
        # 全結合でユニット数を調整して変形
        with tf.variable_scope('reshape'):
            weight0 = tf.get_variable(
                'w', [10, 4 * 4 * 36],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias0 = tf.get_variable(
                'b', shape=[4 * 4 * 36],
                initializer=tf.zeros_initializer()
            )
            fc0 = tf.add(tf.matmul(inputs, weight0), bias0)
            out0 = tf.reshape(fc0, [batch_size, 4, 4, 36])

        # 畳み込みの逆操作を繰り返し
        with tf.variable_scope('conv_transpose1'):
            weight1 = tf.get_variable(
                'w', [5, 5, 24, 36],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias1 = tf.get_variable(
                'b', shape=[24],
                initializer=tf.zeros_initializer()
            )
            deconv1 = tf.nn.conv2d_transpose(out0, weight1, [batch_size, 8, 8, 24], [1, 2, 2, 1])
            out1 = tf.add(deconv1, bias1)
        with tf.variable_scope('conv_transpose2'):
            weight2 = tf.get_variable(
                'w', [5, 5, 16, 24],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias2 = tf.get_variable(
                'b', shape=[16],
                initializer=tf.zeros_initializer()
            )
            deconv2 = tf.nn.conv2d_transpose(out1, weight2, [batch_size, 16, 16, 16], [1, 2, 2, 1])
            out2 = tf.add(deconv2, bias2)
        with tf.variable_scope('conv_transpose3'):
            weight3 = tf.get_variable(
                'w', [5, 5, 1, 16],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias3 = tf.get_variable(
                'b', shape=[1],
                initializer=tf.zeros_initializer()
            )
            deconv3 = tf.nn.conv2d_transpose(out2, weight3, [batch_size, 32, 32, 1], [1, 2, 2, 1])
            out3 = tf.add(deconv3, bias3)
        return tf.nn.tanh(out3)

def discriminator(inputs, reuse=False):
    '''
    入力された画像が学習データのものか否かを識別するモデル
    Args:
        inputs: [batch_size, height(=32), width(=32), channels(=1)]のテンソル
        reuse: 変数を再利用するか否か
    Returns:
        推論結果の [batch_size, 2] のテンソル    
    '''
    with tf.variable_scope('d'):
        # 畳み込みの繰り返し
        with tf.variable_scope('conv1', reuse=reuse):
            weight1 = tf.get_variable(
                'w', [5, 5, 1, 16],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias1 = tf.get_variable(
                'b', shape=[16],
                initializer=tf.zeros_initializer()
            )
            conv1 = tf.nn.conv2d(inputs, weight1, [1, 2, 2, 1], 'SAME')
            out1 = tf.nn.relu(tf.add(conv1, bias1))
        with tf.variable_scope('conv2', reuse=reuse):
            weight2 = tf.get_variable(
                'w', [5, 5, 16, 24],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias2 = tf.get_variable(
                'b', shape=[24],
                initializer=tf.zeros_initializer()
            )
            conv2 = tf.nn.conv2d(out1, weight2, [1, 2, 2, 1], 'SAME')
            out2 = tf.nn.relu(tf.add(conv2, bias2))
        with tf.variable_scope('conv3', reuse=reuse):
            weight3 = tf.get_variable(
                'w', [5, 5, 24, 36],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias3 = tf.get_variable(
                'b', shape=[36],
                initializer=tf.zeros_initializer()
            )
            conv3 = tf.nn.conv2d(out2, weight3, [1, 2, 2, 1], 'SAME')
            out3 = tf.nn.relu(tf.add(conv3, bias3))
        reshape = tf.reshape(out3, [out3.get_shape()[0].value, -1])

        # 全結合で2クラスに分類
        with tf.variable_scope('fully_connect', reuse=reuse):
            weight4 = tf.get_variable(
                'w', [4 * 4 * 36, 2],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            bias4 = tf.get_variable(
                'b', shape=[2],
                initializer=tf.zeros_initializer()
            )
            out4 = tf.add(tf.matmul(reshape, weight4), bias4)
    return out4

batch_size = 32
inputs = tf.random_normal([batch_size, 10])
real = batch(batch_size)
fake = generator(inputs, batch_size)
real_logits = discriminator(real)
fake_logits = discriminator(fake, reuse=True)
g_loss = tf.reduce_mean(tf.nn.sparse)


# end of file