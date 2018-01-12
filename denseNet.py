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
    out_channels = 16
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
