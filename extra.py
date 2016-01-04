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
from __future__ import absolute_import
import numpy as np
from .. import backend as K
from .. import activations, initializations, regularizers, constraints
from ..layers.core import Layer

def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride

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
        return (input_shape[0], input_shape[1], np.prod(input_shape[2:]))

    def get_output(self, train=False):
        X = self.get_input(train)
        finaloutput = K.tdflatten(X)
        return finaloutput

class TimeDistributedConvolution2D(Layer):
    # This layer performs 2D Convolutions with the extra dimension of time
    # Default Input shape (Theano dim ordering): (num_samples, num_timesteps, stack_size, num_rows, num_cols)
    # Deafault Output shape (Theano dim ordering): (num_samples, num_timesteps, num_filters, num_rows, num_cols), Note: num_rows and num_cols could have changed
    # Potential use case: For connecting a Convolutional Layer with a Recurrent or other Time Distributed Layer
    
    input_ndim = 5

    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='th',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, **kwargs):
    
        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for TimeDistributedConvolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        self.input = K.placeholder(ndim=5)
        super(TimeDistributedConvolution2D, self).__init__(**kwargs)

    def build(self):
        if self.dim_ordering == 'th':
            stack_size = self.input_shape[2]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = self.input_shape[4]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.W = self.init(self.W_shape)
        self.b = K.zeros((self.nb_filter,))
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
        if self.dim_ordering == 'th':
            rows = input_shape[3]
            cols = input_shape[4]
        elif self.dim_ordering == 'tf':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])
        if self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], input_shape[1], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        input_dim = self.input_shape
        Y = K.collapsetime(X) #collapse num_samples and num_timesteps
        conv_out = K.conv2d(Y, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering)
        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        finaloutput = K.expandtime(X, output)
        return finaloutput


    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(TimeDistributedConvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class _TimeDistributedPooling2D(Layer):
    '''Abstract class for different Time Distributed pooling 2D layers.
    '''
    input_ndim = 5

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(_TimeDistributedPooling2D, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=5)
        self.pool_size = tuple(pool_size)
        if strides is None:
            strides = self.pool_size
        self.strides = tuple(strides)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            rows = input_shape[3]
            cols = input_shape[4]
        elif self.dim_ordering == 'tf':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.pool_size[0],
                                  self.border_mode, self.strides[0])
        cols = conv_output_length(cols, self.pool_size[1],
                                  self.border_mode, self.strides[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], input_shape[2], rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], input_shape[1], rows, cols, input_shape[4])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        raise NotImplementedError

    def get_output(self, train=False):
        X = self.get_input(train)
        input_dim = self.input_shape
        Y = K.collapsetime(X) #collapse num_samples and num_timesteps
        output = self._pooling_function(inputs=Y, pool_size=self.pool_size,
                                        strides=self.strides,
                                        border_mode=self.border_mode,
                                        dim_ordering=self.dim_ordering)
        finaloutput = K.expandtime(X, output)
        return finaloutput

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'pool_size': self.pool_size,
                  'border_mode': self.border_mode,
                  'strides': self.strides,
                  'dim_ordering': self.dim_ordering}
        base_config = super(_TimeDistributedPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TimeDistributedMaxPooling2D(_TimeDistributedPooling2D):
    # This layer performs 2D Max Pooling with the extra dimension of time
    # Default Input shape (Theano dim ordering): (num_samples, num_timesteps, stack_size, num_rows, num_cols)
    # Default Output shape (Theano dim ordering): (num_samples, num_timesteps, stack_size, new_num_rows, new_num_cols)
    # Potential use case: For stacking after a Time Distributed Convolutional Layer or other Time Distributed Layer
    
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(TimeDistributedMaxPooling2D, self).__init__(pool_size, strides, border_mode,
                                           dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='max')
        return output

class TimeDistributedAveragePooling2D(_TimeDistributedPooling2D):
    # This layer performs 2D Average Pooling with the extra dimension of time
    # Default Input shape (Theano dim ordering): (num_samples, num_timesteps, stack_size, num_rows, num_cols)
    # Default Output shape (Theano dim ordering): (num_samples, num_timesteps, stack_size, new_num_rows, new_num_cols)
    # Potential use case: For stacking after a Time Distributed Convolutional Layer or other Time Distributed Layer

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(TimeDistributedAveragePooling2D, self).__init__(pool_size, strides, border_mode,
                                               dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='avg')
        return output