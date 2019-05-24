import math
import numpy as np
import tensorflow as tf
from dcgan_util import *

gf_dim = 64
df_dim = 64
batch_size = 100

# encode_dim = 100
img_encode_bn = False
img_decode_bn = False
img_disc_bn = False

word_encode_bn = False
word_decode_bn = False
word_disc_bn = False

def leaky_relu(x, leak=0.2, name="leaky_relu"):
    return tf.maximum(x, leak*x)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu), stddev = 1.0)
    res = mu + tf.exp(log_var/2)*eps
    return res

def square_clf( inputs, name = "sub" ):
    with tf.variable_scope( name, reuse = tf.AUTO_REUSE) as scope:
        if img_encode_bn:
            bn_1 = batch_norm(name='img_ebn_1')
            bn_2 = batch_norm(name='img_ebn_2')
            bn_3 = batch_norm(name='img_ebn_3')
            bn_4 = batch_norm(name='img_ebn_4')
        # print("img_encode input shape", inputs.get_shape().as_list())
        if img_encode_bn:
            h0 = tf.nn.relu(bn_1(conv2d(inputs, 32, k_h=3, k_w=3, name='h0_conv')))
        else:
            h0 = tf.nn.relu(conv2d(inputs, 32, k_h=3, k_w=3, name='h0_conv'))
        # print("img_encode h0 shape", h0.get_shape().as_list())
        if img_encode_bn:
            h1 = tf.nn.relu(bn_2(conv2d(h0, 64, k_h=3, k_w=3, name='h1_conv')))
        else:
            h1 = tf.nn.relu(conv2d(h0, 64, k_h=3, k_w=3, name='h1_conv'))
        # print("img_encode h1 shape", h1.get_shape().as_list())
        if img_encode_bn:
            h2 = tf.nn.relu(bn_3(conv2d(h1, 128, k_h=3, k_w=3, name='h2_conv')))
        else:
            h2 = tf.nn.relu(conv2d(h1, 128, k_h=3, k_w=3, name='h2_conv'))
        if img_encode_bn:
            h3 = tf.nn.relu(bn_4(conv2d(h2, 256, k_h=3, k_w=3, name='h3_conv')))
        else:
            h3 = tf.nn.relu(conv2d(h2, 256, k_h=3, k_w=3, name='h3_conv'))
        # print('h2 shape', h2.shape)
        # z_mu_l0 = tf.layers.dense(h1, 256, activation = tf.nn.relu, name='z_mu_l0')
        rs = tf.layers.flatten( h3 )
        return rs

def single_scale_square_clf( inputs, class_size ):
    with tf.variable_scope( "classifier", reuse = tf.AUTO_REUSE ) as scope:
        rs = square_clf( inputs )
        rs = tf.layers.dense( rs, 512, activation = tf.nn.relu, name = "dense_0" )
        rs = tf.layers.dense( rs, 512, activation = tf.nn.relu, name = "dense_0_1" )
        rs = tf.layers.dense( rs, class_size, name = "dense_1" )
        return rs

def pair_scale_square_clf( left_input, left_dim, right_input, right_dim, class_size ):
    with tf.variable_scope( "classifier", reuse = tf.AUTO_REUSE ) as scope:
        rs_left = square_clf( left_input, name = "sub_left" )
        rs_right = square_clf( right_input, name = "sub_right" )
        rs = tf.concat( 1, [ rs_left, rs_right ], name = "concat" )
        rs = tf.layers.dense( rs, 1024, activation = tf.nn.relu, name = "dense_0" )
        rs = tf.layers.dense( rs, class_size, name = "dense_1" )
        return rs

def line_clf( inputs, input_dim, name = "sub" ):
    with tf.variable_scope( name, reuse = tf.AUTO_REUSE ) as scope:
        if img_encode_bn:
            bn_1 = batch_norm(name='img_ebn_1')
        # print("img_encode input shape", inputs.get_shape().as_list())
        if img_encode_bn:
            h0 = tf.nn.relu(bn_1(conv2d(inputs, 256, k_h=input_dim, k_w=1, d_h = input_dim, d_w = 1, name='h0_conv')))
        else:
            h0 = tf.nn.relu(conv2d(inputs, 256, k_h=input_dim, k_w=1, d_h = input_dim, d_w = 1, name='h0_conv'))
        print( "h0", h0.get_shape().as_list() )
        rs = tf.layers.flatten( h0 )
        print( "rs", rs.get_shape().as_list() )
        return rs

def single_line_clf( inputs, input_dim, class_size ):
    with tf.variable_scope( "classifier", reuse = tf.AUTO_REUSE ) as scope:
        rs = line_clf( inputs, input_dim )
        rs = tf.layers.dense( rs, 1024, activation = tf.nn.relu, name = "dense_0" )
        rs = tf.layers.dense( rs, class_size, name = "dense_3" )
        return rs

def pair_scale_line_clf( left_input, left_dim, right_input, right_dim, class_size ):
    with tf.variable_scope( "classifier", reuse = tf.AUTO_REUSE ) as scope:
        rs_left = line_clf( left_input, left_dim, name = "sub_left" )
        rs_right = line_clf( right_input, right_dim, name = "sub_right" )
        rs = tf.concat( 1, [ rs_left, rs_right ], name = "concat" )
        rs = tf.layers.dense( rs, 1024, activation = tf.nn.relu, name = "dense_0" )
        rs = tf.layers.dense( rs, class_size, name = "dense_1" )
        return rs

def img_clf(inputs, encode_dim):
    with tf.variable_scope('img_encoder', reuse = tf.AUTO_REUSE) as scope:
        bn_1 = batch_norm(name='img_ebn_1')
        bn_2 = batch_norm(name='img_ebn_2')
        bn_3 = batch_norm(name='img_ebn_3')
        bn_4 = batch_norm(name='img_ebn_4')
        # print("img_encode input shape", inputs.get_shape().as_list())
        if img_encode_bn:
            h0 = tf.nn.relu(bn_1(conv2d(inputs, 64, k_h=3, k_w=3, name='h0_conv')))
        else:
            h0 = tf.nn.relu(conv2d(inputs, 64, k_h=3, k_w=3, name='h0_conv'))
        # print("img_encode h0 shape", h0.get_shape().as_list())
        if img_encode_bn:
            h1 = tf.nn.relu(bn_2(conv2d(h0, 32, k_h=3, k_w=3, name='h1_conv')))
        else:
            h1 = tf.nn.relu(conv2d(h0, 32, k_h=3, k_w=3, name='h1_conv'))
        # print("img_encode h1 shape", h1.get_shape().as_list())
        if img_encode_bn:
            h2 = tf.nn.relu(bn_3(conv2d(h1, 8, k_h=3, k_w=3, name='h2_conv')))
        else:
            h2 = tf.nn.relu(conv2d(h1, 8, k_h=3, k_w=3, name='h2_conv'))
        if img_encode_bn:
            h3 = tf.nn.relu(bn_4(conv2d(h2, 4, k_h=3, k_w=3, name='h3_conv')))
        else:
            h3 = tf.nn.relu(conv2d(h2, 4, k_h=3, k_w=3, name='h3_conv'))
        # print('h2 shape', h2.shape)
        # z_mu_l0 = tf.layers.dense(h1, 256, activation = tf.nn.relu, name='z_mu_l0')
        rs = tf.layers.flatten(h3)
        # print("rs shape", rs.get_shape().as_list())
        z_mu_l0 = tf.nn.relu(linear(rs, 1024, 'z_mu_l0'))
        z_mu_l1 = tf.layers.dense(z_mu_l0, encode_dim, activation = None, name='z_mu_l1')
        return z_mu_l1