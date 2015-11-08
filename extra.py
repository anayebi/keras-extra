# Extra Layers that I have added to Keras
# Layers that have been added to the Keras master branch will be noted both in the ReadMe and removed from extra.py.
#
# Copyright Aran Nayebi, 2015
# anayebi@stanford.edu
#
# If you already have Keras installed, for this to work on your current installation, please do the following:
# 1. Upgrade to the newest version of Keras (since some layers may have been added from here that are now commented out):
#    sudo pip install --upgrade git+git://github.com/fchollet/keras.git
# or, if you don't have super user access, just run:
#    pip install --upgrade git+git://github.com/fchollet/keras.git --user
#
# 2. Add this file to your Keras installation in the layers directory (keras/layers/)
#
# 3. Now, to use any layer, just run:
#    from keras.layers.extra import layername
#
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import numpy as np

from .. import activations, initializations, regularizers, constraints
from ..utils.theano_utils import shared_zeros, floatX, on_gpu
from ..utils.generic_utils import make_tuple
from ..regularizers import ActivityRegularizer, Regularizer

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip
srng = RandomStreams(seed=np.random.randint(10e6))

from ..layers.core import Layer

if on_gpu():
    from theano.sandbox.cuda import dnn

def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'full', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'full':
        output_length = input_length + filter_size - 1
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride

def pool_output_length(input_length, pool_size, ignore_border, stride):
    if input_length is None:
        return None
    if ignore_border:
        output_length = input_length - pool_size + 1
        output_length = (output_length + stride - 1) // stride
    else:
        if pool_size == input_length:
            output_length = min(input_length, stride - stride % 2)
            if output_length <= 0:
                output_length = 1
        elif stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = (input_length - pool_size + stride - 1) // stride
            if output_length <= 0:
                output_length = 1
            else:
                output_length += 1
    return output_length

class TimeDistributedFlatten(Layer):
    # This layer reshapes input to be flat across timesteps (cannot be used as the first layer of a model)
    # Input shape: (num_samples, num_timesteps, *)
    # Output shape: (num_samples, num_timesteps, num_input_units)
    # Potential use case: For stacking after a Time Distributed Convolution/Max Pooling Layer or other Time Distributed Layer
    def __init__(self, **kwargs):
        super(TimeDistributedFlatten, self).__init__(**kwargs)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[0], np.prod(input_shape[2:]))

    def get_output(self, train=False):
        X = self.get_input(train)
        size = theano.tensor.prod(X[0].shape) // X[0].shape[0]
        nshape = (X.shape[0], X.shape[1], size)
        return theano.tensor.reshape(X, nshape)

class TimeDistributedConvolution2D(Layer):
    # This layer performs 2D Convolutions with the extra dimension of time
    # Input shape: (num_samples, num_timesteps, stack_size, num_rows, num_cols)
    # Output shape: (num_samples, num_timesteps, num_filters, num_rows, num_cols), Note: num_rows and num_cols could have changed
    # Potential use case: For connecting a Convolutional Layer with a Recurrent or other Time Distributed Layer
    
    input_ndim = 5

    def __init__(self, nb_filter, nb_row, nb_col,
        init='glorot_uniform', activation='linear', weights=None,
        border_mode='valid', subsample=(1, 1),
        W_regularizer=None, b_regularizer=None, activity_regularizer=None, 
        W_constraint=None, b_constraint=None, **kwargs):
    
        if border_mode not in {'valid', 'full', 'same'}:
            raise Exception('Invalid border mode for TimeDistributedConvolution2D:', border_mode)

        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple(subsample)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        super(TimeDistributedConvolution2D,self).__init__(**kwargs)

    def build(self):
        stack_size = self.input_shape[2]
        dtensor5 = T.TensorType('float32', (False,)*5)
        self.input = dtensor5()
        self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((self.nb_filter,))

        self.params = [self.W, self.b]

        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        rows = input_shape[3]
        cols = input_shape[4]
        rows = conv_output_length(rows, self.nb_row, self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col, self.border_mode, self.subsample[1])
        return (input_shape[0], input_shape[1], self.nb_filter, rows, cols)

    def get_output(self, train=False):
        X = self.get_input(train)
        newshape = (X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])
        Y = theano.tensor.reshape(X, newshape) #collapse num_samples and num_timesteps
        border_mode = self.border_mode
        if on_gpu() and dnn.dnn_available():
            if border_mode == 'same':
                assert(self.subsample == (1, 1))
                pad_x = (self.nb_row - self.subsample[0]) // 2
                pad_y = (self.nb_col - self.subsample[1]) // 2
                conv_out = dnn.dnn_conv(img=Y,
                                        kerns=self.W,
                                        border_mode=(pad_x, pad_y))
            else:
                conv_out = dnn.dnn_conv(img=Y,
                                        kerns=self.W,
                                        border_mode=border_mode,
                                        subsample=self.subsample)
        else:
            if border_mode == 'same':
                border_mode = 'full'

            conv_out = theano.tensor.nnet.conv.conv2d(Y, self.W,
                border_mode=border_mode, subsample=self.subsample)

            if self.border_mode == 'same':
                shift_x = (self.nb_row - 1) // 2
                shift_y = (self.nb_col - 1) // 2
                conv_out = conv_out[:, :, shift_x:Y.shape[2] + shift_x, shift_y:Y.shape[3] + shift_y]

        output = self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        newshape = (X.shape[0], X.shape[1], output.shape[1], output.shape[2], output.shape[3])
        return theano.tensor.reshape(output, newshape)


    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "nb_filter": self.nb_filter,
                  "nb_row": self.nb_row,
                  "nb_col": self.nb_col,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "border_mode": self.border_mode,
                  "subsample": self.subsample,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(TimeDistributedConvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeDistributedMaxPooling2D(Layer):
    # This layer performs 2D Max Pooling with the extra dimension of time
    # Input shape: (num_samples, num_timesteps, stack_size, num_rows, num_cols)
    # Output shape: (num_samples, num_timesteps, stack_size, new_num_rows, new_num_cols)
    # Potential use case: For stacking after a Time Distributed Convolutional Layer or other Time Distributed Layer
    
    input_ndim = 5

    def __init__(self, pool_size=(2, 2), stride=None, ignore_border=True, **kwargs):
        super(TimeDistributedMaxPooling2D,self).__init__(**kwargs)
        dtensor5 = T.TensorType('float32', (False,)*5)
        self.input = dtensor5()
        self.pool_size = tuple(pool_size)
        if stride is None:
            stride = self.pool_size
        self.stride = tuple(stride)
        self.ignore_border = ignore_border

    @property
    def output_shape(self):
        input_shape = self.input_shape
        rows = pool_output_length(input_shape[3], self.pool_size[0], self.ignore_border, self.stride[0])
        cols = pool_output_length(input_shape[4], self.pool_size[1], self.ignore_border, self.stride[1])
        return (input_shape[0], input_shape[1], input_shape[2], rows, cols)

    def get_output(self, train):
        X = self.get_input(train)
        newshape = (X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])
        Y = theano.tensor.reshape(X, newshape) #collapse num_samples and num_timesteps
        output = downsample.max_pool_2d(Y, ds=self.pool_size, st=self.stride, ignore_border=self.ignore_border)
        newshape = (X.shape[0], X.shape[1], output.shape[1], output.shape[2], output.shape[3])
        return theano.tensor.reshape(output, newshape) #shape is (num_samples, num_timesteps, stack_size, new_nb_row, new_nb_col)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "pool_size": self.pool_size,
                  "ignore_border": self.ignore_border,
                  "stride": self.stride}
        base_config = super(TimeDistributedMaxPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
