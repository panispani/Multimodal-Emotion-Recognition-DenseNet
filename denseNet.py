import tensorflow as tf

def dropout(inputs, is_training, dropout_rate):
    """ Perform dropout if we are training according to the given `dropout_rate`
    """
    if dropout_rate == 0:
        return inputs

    if is_training:
        return tf.layers.dropout(inputs, dropout_rate)
    else:
        return inputs

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

def composite_nonlinearfunction(inputs, out_channels, is_training, kernel_size, dropout_rate):
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
        output = convolution2d(output, kernel_size, out_channels)
        output = dropout(output, is_training, dropout_rate)
    return output

def add_transition_layer_to_classes(inputs, is_training, pool_size):
    """Last transition to get to classes
    - BN
    - ReLU
    - Global average pooling
    - FC layer multiplication - This is done outside the model?
    """
    # BN
    output = tf.contrib.layers.batch_norm(
                inputs, scale=True, is_training=is_training, updates_collections=None)
    # ReLU
    output = tf.nn.relu(output)

    # Global average pooling
    # In the official implementation the pool_size is set to 7 or 8 according to the dataset
    window = [1, pool_size, pool_size, 1]
    strides = [1, pool_size, pool_size, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool(output, window, strides, padding)

    return output

def add_transition_layer(inputs, is_training, dropout_rate, reduction):
    """Perform BN + ReLU + conv2D with 1x1 kernel + 2x2 avg pooling
       ReLU is not specified in the paper but it's included in the official implementation
    """
    # reduction rate is 1 if compression is not performed
    out_features = int(int(inputs.get_shape()[-1]) * reduction)
    output = composite_nonlinearfunction(inputs, out_features, is_training, 1, dropout_rate)

    # 2 x 2 average pooling
    window = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool(output, window, strides, padding)
    return output

def bottleneck(inputs, growth_rate, is_training, dropout_rate):
    """Perform bottlenck by doing BN + ReLU + conv2D with
       1x1 kernel and 4 * `growth_rate` features
    """
    with tf.variable_scope("Bottleneck"):
        output = tf.contrib.layers.batch_norm(
                     inputs, scale=True, is_training=is_training, updates_collections=None)
        output = tf.nn.relu(output)
        out_features = 4 * growth_rate
        kernel_size = 1
        output = convolution2d(output, kernel_size, out_features)
        output = dropout(output, is_training, dropout_rate)
    return output

def add_internal_layer(inputs, growth_rate, is_training, dropout_rate, b_mode):
    """Perform H_l composite function, with the given `kernel_size`, for the layer
       and afterwards concatenate input with output from composite function.
    """
    if b_mode:
        output = bottleneck(inputs, growth_rate, is_training, dropout_rate)
    else:
        output = inputs

    kernel_size = 3
    output = composite_nonlinearfunction(output, growth_rate, is_training, kernel_size, dropout_rate)
    # TODO axis is the dimension along which to concatate, I think this will have to change if we don't work with images
    # This is the issue of high memory usage, here we want shared buffers, not yet implemented in Tensorflow yet
    # See relevant issue in repo: https://github.com/tensorflow/tensorflow/issues/12948
    output = tf.concat(axis=3, values=(inputs, output))
    return output

def denseNet(inputs,
             num_classes=None,
             is_training=True,
             total_blocks=3,
             depth=40,
             growth_rate=12,
             dropout_rate=0,   # 0.2
             b_mode=False,
             reduction=1,      # 0.5
             pool_size=7):
    """Creates the densnet model.
    Args:
        inputs: A tensor that contains the input.
        num_classes : Number of predicted classes for classification tasks
        is_training : if the model is in training mode
        total_blocks: total dense blocks in the DenseNet
        depth       : number of layers in the DenseNet
        growth_rate : growth rate in the DenseNet
        dropout_rate: rate of dropout, when set to 0 there is no dropout performed
        b_mode      : whether bottleneck is used
        reduction   : compression factor > 0 and <= 1 (equality in case of no reduction)
        pool_size   : transitioning to classes global average pooling size
    Returns:
        The densenet model.
    """
    # out_features is 16 or 2 * growth_rate for DenseNet-BC
    out_features = 2 * growth_rate

    # The initial transformation is also dependent on the dataset.
    # 3x3 convolution and 7x7 convolution (Stride 2, Padding 3) + BN + ReLU + 3x3 maxpooling are used(2 stride, 3 padding) are used
    with tf.variable_scope("Initial_Convolution"):
        initial_conv_size = 3
        densenet = convolution2d(inputs, initial_conv_size, out_features)

    # Add 'total_blocks' blocks to the densenet
    layers_per_block = int((depth - (total_blocks + 1)) / total_blocks)
    if b_mode:
        layers_per_block = layers_per_block // 2

    for i in range(total_blocks):
        # If this is not the first block to be added, add a transition layer before it
        if i != 0:
            with tf.variable_scope("Transition_Layer_" + str(i)):
                densenet = add_transition_layer(densenet, is_training, dropout_rate, reduction)

        with tf.variable_scope("Block_" + str(i)):
            for j in range(layers_per_block):
                with tf.variable_scope("Inner_Layer_" + str(j)):
                    densenet = add_internal_layer(densenet, growth_rate, is_training, dropout_rate, b_mode)

    with tf.variable_scope("Transition_layer_to_classes"):
        densenet = add_transition_layer_to_classes(densenet, is_training, pool_size)
    return densenet
