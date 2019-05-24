import tensorflow as tf
import numpy as np
import os
from models import *

class CLF( object ):

    def __init__( self, opts ):
        self.sess = tf.Session()
        self.opts = opts
        self.add_input_placeholder()
        self.construct_clf()
        self.construct_loss()
        self.construct_optimizer()
        self.init()
        self.saver = tf.train.Saver( max_to_keep = 100)

    def init( self ):
        self.sess.run( [ tf.global_variables_initializer() ] )

    def add_input_placeholder( self ):
        self.s_input_sample = tf.placeholder( tf.float32, [ self.opts.batch_size ] + self.opts.s_data_source.input_shape, name = "s_input_sample" )
        self.s_input_label = tf.placeholder( tf.float32, [ self.opts.batch_size, self.opts.class_size ], name = "s_input_label" )
        
        if self.opts.t_sample_path is not None:
            self.t_input_sample = tf.placeholder( tf.float32, [ self.opts.batch_size ] + self.opts.t_data_source.input_shape, name = "t_input_sample" )
            self.t_input_label = tf.placeholder( tf.float32, [ self.opts.batch_size, self.opts.class_size ], name = "t_input_label" )

    def construct_clf( self ):
        if self.opts.clf_type == "line":
            if self.opts.t_sample_path is not None:
                self.res = pair_scale_line_clf( self.s_input_sample, self.opts.s_dim, self.t_input_sample, self.opts.t_dim, self.opts.class_size )
            else:
                self.res = single_line_clf( self.s_input_sample, self.opts.s_dim, self.opts.class_size )
        elif self.opts.clf_type == "square":
            if self.opts.t_sample_path is not None:
                self.res = pair_scale_square_clf( self.s_input_sample, self.opts.s_dim, self.t_input_sample, self.opts.t_dim, self.opts.class_size )
            else:
                self.res = single_scale_square_clf( self.s_input_sample, self.opts.class_size )
        else:
            raise NotImplementedError()

        # self.activated_res =  tf.nn.sigmoid( self.res )
        self.activated_res =  tf.exp( self.res )
        

    def construct_loss( self ):
        self.input_label = tf.exp( self.s_input_label )
        self.mse = tf.reduce_mean( tf.square( self.activated_res - self.input_label ) )
        self.loss = tf.reduce_mean( self.mse )
        # self.loss = tf.nn.sigmoid_cross_entropy_with_logits( labels = self.input )
        
        # self.loss = tf.reduce_mean( self.mse )

    def construct_optimizer( self ):
        if self.opts.optimizer == "adam":
            vae_optimizer = tf.train.AdamOptimizer( self.opts.lr, beta1=0.9, beta2 = 0.99 )
        elif self.opts.optimizer == "sgd":
            vae_optimizer = tf.train.GradientDescentOptimizer( self.opts.vae_lr )
        self.vae_optimizer = vae_optimizer.minimize( loss = self.loss )
        

    def train( self ):
        self.loss_list = []
        self.mse_list = []
        self.precision_list = []
        for i in range( self.opts.train_iter + 1 ):
            if self.opts.t_sample_path is not None: 
                s_input_sample, s_input_label = self.opts.s_data_source.next_batch()
                t_input_sample, t_input_label = self.opts.t_data_source.next_batch()
                self.sess.run( self.vae_optimizer, feed_dict = { self.s_input_sample : s_input_sample, self.s_input_label: s_input_label,
                                                                 self.t_input_sample: t_input_sample, self.t_input_label: t_input_label } )
                if i % 100 == 0:
                    print( "Iteration",i )
                    mse = np.mean( self.sess.run( self.mse, feed_dict = { self.s_input_sample : s_input_sample, self.s_input_label: s_input_label,
                                                                 self.t_input_sample: t_input_sample, self.t_input_label: t_input_label } ) )
                    print( "MSE: ", mse )
                    self.mse_list.append( mse )
                    loss = np.mean( self.sess.run( self.loss, feed_dict = { self.s_input_sample : s_input_sample, self.s_input_label: s_input_label,
                                                                 self.t_input_sample: t_input_sample, self.t_input_label: t_input_label } ) )
                    print( "LOSS: ", loss )
                    self.loss_list.append( loss )
                    print("-----------------")
            else:
                input_sample, input_label = self.opts.s_data_source.next_batch()
                self.sess.run( self.vae_optimizer, feed_dict = { self.s_input_sample : input_sample, self.s_input_label: input_label } )
                if i % 100 == 0:

                    mse = np.mean( self.sess.run( self.mse, feed_dict = { self.s_input_sample : input_sample, self.s_input_label: input_label } ) )
                    print( "MSE: ", mse )
                    self.mse_list.append( mse )
                    loss = np.mean( self.sess.run( self.loss, feed_dict = { self.s_input_sample : input_sample, self.s_input_label: input_label } ) )
                    print( "LOSS: ", loss )
                    self.loss_list.append( loss )
                    print("-----------------")

            if i != 0 and i % 20000 == 0:
                path = self.opts.cpt_path +"/"+ str( i )
                os.mkdir( path )
                path += "/model.ckpt"
                self.saver.save( self.sess, path )
