# keras-extra
Extra Layers that I have added to Keras

Layers that have been added to the Keras master branch will be noted both in the ReadMe and in extra.py. Feel free to use that layer either from Keras directly or from the version in extra.py (note I will no longer update a layer in extra.py once it has been added to Keras). 

Aran Nayebi, 2015

anayebi@stanford.edu

# Installation Instructions
If you already have Keras installed, for this to work on your current installation, please do the following:

1. Upgrade to the newest version of Keras (since some layers may have been added from here that are now commented out):
    
    sudo pip install --upgrade git+git://github.com/fchollet/keras.git

or, if you don't have super user access, just run:
    
    pip install --upgrade git+git://github.com/fchollet/keras.git --user

2. Add extra.py to your Keras installation in the layers directory (keras/layers/)

3. Now, to use any layer, just run:
    
    from keras.layers.extra import layername

# Layers

- **Permute** (added to keras.layers.core, July 17 2015: https://github.com/fchollet/keras/pull/409)

    Permutes the dimensions of the data according to the given tuple.
    
    Input shape: This layer does not assume a specific input shape.
    
    Output shape: Same as the input shape, but with the dimensions re-ordered according to the ordering specified by the tuple.

    Arguments: Tuple is a tensor that specifies the ordering of the dimensions of the data.

- **TimeDistributedFlatten**

	This layer reshapes input to be flat across timesteps (cannot be used as the first layer of a model)

	Input shape: (num_samples, num_timesteps, *)
	
	Output shape: (num_samples, num_timesteps, num_input_units)
	
	Potential use case: For stacking after a Time Distributed Convolution/Max Pooling Layer or other Time Distributed Layer
	
- **TimeDistributedConvolution2D**

	This layer performs 2D Convolutions with the extra dimension of time
	
    Input shape: (num_samples, num_timesteps, stack_size, num_rows, num_cols)
	
    Output shape: (num_samples, num_timesteps, num_filters, num_rows, num_cols), Note: num_rows and num_cols could have changed
	
    Potential use case: For connecting a Convolutional Layer with a Recurrent or other Time Distributed Layer

- **TimeDistributedMaxPooling2D**

    This layer performs 2D Max Pooling with the extra dimension of time
	
    Input shape: (num_samples, num_timesteps, stack_size, num_rows, num_cols)
	
    Output shape: (num_samples, num_timesteps, stack_size, new_num_rows, new_num_cols)
	
    Potential use case: For stacking after a Time Distributed Convolutional Layer or other Time Distributed Layer

- **UpSample1D** (added to keras.layers.convolutional, August 16 2015: https://github.com/fchollet/keras/pull/532)

	This layer upsamples input across one dimension (e.g. inverse MaxPooling1D)
	
    Input shape: (num_samples, steps, dim)
	
    Output shape: (num_samples, upsampled_steps, dim)
	
    Potential use case: For stacking after a MaxPooling1D Layer

- **UpSample2D** (added to keras.layers.convolutional, August 16 2015: https://github.com/fchollet/keras/pull/532)

	This layer upsamples input across two dimensions (e.g. inverse MaxPooling2D)
	
    Input shape: (num_samples, stack_size, num_rows, num_cols)
	
    Output shape: (num_samples, stack_size, new_num_rows, new_num_cols)
	
    Potential use case: For stacking after a MaxPooling2D Layer

- **Dense2D**

	This layer performs an affine transformation on a 2D input
	
    Input shape: (num_samples, input_dim_rows, input_dim_cols)
	
    Output shape: (num_samples, input_dim_rows, output_dim_cols)
	
    Potential use case: For layer L, does LW + b, where W is input_dim_cols x output_dim_cols weight matrix and b is input_dim_rows x output_dim_cols bias
