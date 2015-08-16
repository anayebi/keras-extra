# Extra Layers that I have added to Keras
# Layers that have been added to the Keras master branch will be noted both in the ReadMe and in extra.py. 
# Feel free to use that layer either from Keras directly or from the version in extra.py 
# (note I will no longer update a layer in extra.py once it has been added to Keras).
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
from ..utils.theano_utils import shared_zeros, floatX
from ..utils.generic_utils import make_tuple
from ..regularizers import ActivityRegularizer, Regularizer

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip
srng = RandomStreams(seed=np.random.randint(10e6))

from ..layers.core import Layer

# Permute Layer added to keras.layers.core, July 17 2015
class Permute(Layer):
	# Permutes the dimensions of the data according to the given tuple.
	# Input shape: This layer does not assume a specific input shape.
	# Output shape: Same as the input shape, but with the dimensions re-ordered according to the ordering specified by the tuple.
	# Arguments:
	# Tuple is a tensor that specifies the ordering of the dimensions of the data.
	def __init__(self, dims):
		super(Permute,self).__init__()
		self.dims = dims

	def get_output(self, train):
		X = self.get_input(train)
		return X.dimshuffle((0,) + self.dims)

	def get_config(self):
		return {"name":self.__class__.__name__,
			"dims":self.dims}

class TimeDistributedFlatten(Layer):
	# This layer reshapes input to be flat across timesteps (cannot be used as the first layer of a model)
	# Input shape: (num_samples, num_timesteps, *)
	# Output shape: (num_samples, num_timesteps, num_input_units)
	# Potential use case: For stacking after a Time Distributed Convolution/Max Pooling Layer or other Time Distributed Layer
	def __init__(self):
		super(TimeDistributedFlatten, self).__init__()

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
	def __init__(self, nb_filter, stack_size, nb_row, nb_col,
		init='glorot_uniform', activation='linear', weights=None,
		border_mode='valid', subsample=(1, 1),
		W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None):
	
		if border_mode not in {'valid', 'full', 'same'}:
			raise Exception('Invalid border mode for TimeDistributedConvolution2D:', border_mode)

		super(TimeDistributedConvolution2D,self).__init__()
		self.init = initializations.get(init)
		self.activation = activations.get(activation)
		self.subsample = subsample
		self.border_mode = border_mode
		self.nb_filter = nb_filter
		self.stack_size = stack_size

		self.nb_row = nb_row
		self.nb_col = nb_col
		dtensor5 = T.TensorType('float32', (False,)*5)
		self.input = dtensor5()
		self.W_shape = (nb_filter, stack_size, nb_row, nb_col)
		self.W = self.init(self.W_shape)
		self.b = shared_zeros((nb_filter,))

		self.params = [self.W, self.b]

		self.regularizers = []

		self.W_regularizer = regularizers.get(W_regularizer)
		if self.W_regularizer:
			self.W_regularizer.set_param(self.W)
			self.regularizers.append(self.W_regularizer)

		self.b_regularizer = regularizers.get(b_regularizer)
		if self.b_regularizer:
			self.b_regularizer.set_param(self.b)
			self.regularizers.append(self.b_regularizer)

		self.activity_regularizer = regularizers.get(activity_regularizer)
		if self.activity_regularizer:
			self.activity_regularizer.set_layer(self)
			self.regularizers.append(self.activity_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)
		self.constraints = [self.W_constraint, self.b_constraint]

		if weights is not None:
			self.set_weights(weights)

	def get_output(self, train):
		X = self.get_input(train)
		newshape = (X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])
		Y = theano.tensor.reshape(X, newshape) #collapse num_samples and num_timesteps
		border_mode = self.border_mode
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
		return {"name":self.__class__.__name__,
			"nb_filter":self.nb_filter,
			"stack_size":self.stack_size,
			"nb_row":self.nb_row,
			"nb_col":self.nb_col,
			"init":self.init.__name__,
			"activation":self.activation.__name__,
			"border_mode":self.border_mode,
			"subsample":self.subsample,
			"W_regularizer":self.W_regularizer.get_config() if self.W_regularizer else None,
			"b_regularizer":self.b_regularizer.get_config() if self.b_regularizer else None,
			"activity_regularizer":self.activity_regularizer.get_config() if self.activity_regularizer else None,
			"W_constraint":self.W_constraint.get_config() if self.W_constraint else None,
			"b_constraint":self.b_constraint.get_config() if self.b_constraint else None}

class TimeDistributedMaxPooling2D(Layer):
	# This layer performs 2D Max Pooling with the extra dimension of time
	# Input shape: (num_samples, num_timesteps, stack_size, num_rows, num_cols)
	# Output shape: (num_samples, num_timesteps, stack_size, new_num_rows, new_num_cols)
	# Potential use case: For stacking after a Time Distributed Convolutional Layer or other Time Distributed Layer
	def __init__(self, poolsize=(2, 2), stride=None, ignore_border=True):
		super(TimeDistributedMaxPooling2D,self).__init__()
		dtensor5 = T.TensorType('float32', (False,)*5)
		self.input = dtensor5()
		self.poolsize = poolsize
		self.stride = stride
		self.ignore_border = ignore_border


	def get_output(self, train):
		X = self.get_input(train)
		newshape = (X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])
		Y = theano.tensor.reshape(X, newshape) #collapse num_samples and num_timesteps
		output = downsample.max_pool_2d(Y, ds=self.poolsize, st=self.stride, ignore_border=self.ignore_border)
		newshape = (X.shape[0], X.shape[1], output.shape[1], output.shape[2], output.shape[3])
		return theano.tensor.reshape(output, newshape) #shape is (num_samples, num_timesteps, stack_size, new_nb_row, new_nb_col)

	def get_config(self):
		return {"name":self.__class__.__name__,
				"poolsize":self.poolsize,
				"ignore_border":self.ignore_border,
				"stride": self.stride}

# UpSample1D Layer added to keras.layers.convolutional, August 16 2015
class UpSample1D(Layer):
	# This layer upsamples input across one dimension (e.g. inverse MaxPooling1D)
	# Input shape: (num_samples, steps, dim)
	# Output shape: (num_samples, upsampled_steps, dim)
	# Potential use case: For stacking after a MaxPooling1D Layer
	def __init__(self, upsample_length=2):
		super(UpSample1D,self).__init__()
		self.upsample_length = upsample_length
		self.input = T.tensor3()

	def get_output(self, train):
		X = self.get_input(train)
		output = theano.tensor.extra_ops.repeat(X, self.upsample_length, axis=1)
		return output

	def get_config(self):
		return {"name":self.__class__.__name__,
				"upsample_length":self.upsample_length}

# UpSample2D Layer added to keras.layers.convolutional, August 16 2015
class UpSample2D(Layer):
	# This layer upsamples input across two dimensions (e.g. inverse MaxPooling2D)
	# Input shape: (num_samples, stack_size, num_rows, num_cols)
	# Output shape: (num_samples, stack_size, new_num_rows, new_num_cols)
	# Potential use case: For stacking after a MaxPooling2D Layer
	def __init__(self, upsample_size=(2, 2)):
		super(UpSample2D,self).__init__()
		self.input = T.tensor4()
		self.upsample_size = upsample_size


	def get_output(self, train):
		X = self.get_input(train)
		Y = theano.tensor.extra_ops.repeat(X, self.upsample_size[0], axis = 2)
		output = theano.tensor.extra_ops.repeat(Y, self.upsample_size[1], axis = 3)
		return output

	def get_config(self):
		return {"name":self.__class__.__name__,
				"upsample_size":self.upsample_size}

class Dense2D(Layer):
	# This layer performs an affine transformation on a 2D input
	# Input shape: (num_samples, input_dim_rows, input_dim_cols)
	# Output shape: (num_samples, input_dim_rows, output_dim_cols)
	# Potential use case: For layer L, does LW + b, where W is input_dim_cols x output_dim_cols weight matrix and b is input_dim_rows x output_dim_cols bias
	def __init__(self, input_dim_rows, input_dim_cols, output_dim_cols, init='glorot_uniform', activation='linear', weights=None, name=None,
		W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None):

		super(Dense2D, self).__init__()
		self.init = initializations.get(init)
		self.activation = activations.get(activation)
		self.input_dim_rows = input_dim_rows
		self.input_dim_cols = input_dim_cols
		self.output_dim_cols = output_dim_cols

		self.input = T.tensor3()
		self.W = self.init((self.input_dim_cols, self.output_dim_cols))
		self.b = shared_zeros((self.input_dim_rows, self.output_dim_cols))

		self.params = [self.W, self.b]

		self.regularizers = []
		
		self.W_regularizer = regularizers.get(W_regularizer)
		if self.W_regularizer:
			self.W_regularizer.set_param(self.W)
			self.regularizers.append(self.W_regularizer)

		self.b_regularizer = regularizers.get(b_regularizer)
		if self.b_regularizer:
			self.b_regularizer.set_param(self.b)
			self.regularizers.append(self.b_regularizer)

		self.activity_regularizer = regularizers.get(activity_regularizer)
		if self.activity_regularizer:
			self.activity_regularizer.set_layer(self)
			self.regularizers.append(self.activity_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)
		self.constraints = [self.W_constraint, self.b_constraint]

		if weights is not None:
			self.set_weights(weights)

		if name is not None:
			self.set_name(name)

	def set_name(self, name):
		self.W.name = '%s_W' % name
		self.b.name = '%s_b' % name

	def get_output(self, train=False):
		X = self.get_input(train)
		output = self.activation(T.dot(X, self.W) + self.b)
		return output

	def get_config(self):
		return {"name":self.__class__.__name__,
			"input_dim_rows":self.input_dim_rows,
			"input_dim_cols":self.input_dim_cols,
			"output_dim_cols":self.output_dim_cols,
			"init":self.init.__name__,
			"activation":self.activation.__name__,
			"W_regularizer":self.W_regularizer.get_config() if self.W_regularizer else None,
			"b_regularizer":self.b_regularizer.get_config() if self.b_regularizer else None,
			"activity_regularizer":self.activity_regularizer.get_config() if self.activity_regularizer else None,
			"W_constraint":self.W_constraint.get_config() if self.W_constraint else None,
			"b_constraint":self.b_constraint.get_config() if self.b_constraint else None}
