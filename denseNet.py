import tensorflow as tf


# NOTE: As of now this implementation does not use Compression or Bottleneck.

# This can be set to 0 and dropout will be removed
drop_rate = 0.2

def dropout(inputs, is_training):
    if drop_rate == 0:
        return inputs

    if is_training:
        output = tf.layers.dropout(inputs, drop_rate)
    else:
        output = inputs
    return output

def convolution2d(inputs, kernel_size, out_channels):
    """ 2D convolution with stride 1 and given kernel_size and output channels

    Args:
        inputs: A tensor that contains the input, input volume
        kernel_size: size of the kernel
        out_channels: number of output channels (features)
    Returns:
        Tensor containing output volume.
    """
    strides = [1, 1, 1, 1]
    in_channels = int(inputs.get_shape()[-1])
    kernel = tf.get_variable(
      name='kernel',
      shape=[kernel_size, kernel_size, in_channels, out_channels],
      initializer=tf.contrib.layers.variance_scaling_initializer())
    inputs = tf.nn.conv2d(inputs, kernel, strides, 'SAME')
    return inputs

def composite_nonlinearfunction(inputs, out_channels, is_training, kernel_size=3):
    """ H_l function from paper, applied between layers within the dense blocks
        Our function is therefore a composition of:
        - BN
        - ReLU
        - Convolution 3x3, kernel size is 3 with stride [1, 1, 1, 1]
        - When using a dataset without data augmentation:
          Add Dropout with rate 0.2 after each convolutional layer except the first one (before 1st dense block)
    """
    with tf.variable_scope("composite_nonlinearfunction"):
        output = tf.contrib.layers.batch_norm(
                     inputs, scale=True, is_training=is_training, updates_collections=None)
        output = tf.nn.relu(output)
        # If Bottleneck was used here we would firstly use 1x1 convolution
        output = convolution2d(output, kernel_size, out_channels)

        output = dropout(output, is_training)
    return output

def add_internal_layer(inputs, growth_rate, is_training):
   """Perform H_l composite function for the layer and after concatenate
      input with output from composite function.
   """
   # kernel size is 3
   # If we used Bottleneck we would include a 1x1 convolution here first
   output = composite_nonlinearfunction(inputs, growth_rate, is_training, 3)
   # TODO axis is the dimension along which to concatate, I think this will have to change if we don't work with images
   output = tf.concat(axis=3, values=(inputs, output))
   return output

def denseNet(inputs,
             num_classes=None,
             is_training=True,
             depth=40,
             growth_rate=12,
             total_blocks=3):

    """Creates the densnet model.
    Args:
        inputs: A tensor that contains the input.
        num_classes: Number of predicted classes for classification tasks
        is_training : if the model is in training mode
        depth       : number of layers in the DenseNet
        growth_rate : growth rate in the DenseNet
        total_blocks: total dense blocks in the DenseNet
    Returns:
        The densenet model.
    """
    # out_features is 16 or 2 * growth_rate for DenseNet-BC
    # kernel size is 3 for the first convolution
    out_features = 2 * growth_rate
    with tf.variable_scope("Initial_Convolution"):
        densenet = convolution2d(inputs, 3, out_features)

    # Add 'total_blocks' blocks to the densenet
    layers_per_block = int((depth - (total_blocks + 1)) / total_blocks)

    for i in range(total_blocks):
        # If this is not the first block to be added, add a transition layer before it
        if i != 0:
            with tf.variable_scope("Transition_Layer_" + str(i)):
                densenet = add_transition_layer(densenet, is_training)

        with tf.variable_scope("Block_" + str(i)):
            for j in range(layers_per_block):
                with tf.variable_scope("Inner_Layer_" + str(j)):
                    densenet = add_internal_layer(densenet, growth_rate, is_training)

    with tf.variable_scope("Transition_layer_to_classes"):
        densenet = add_transition_layer_to_classes(densenet, is_training)
    return densenet
