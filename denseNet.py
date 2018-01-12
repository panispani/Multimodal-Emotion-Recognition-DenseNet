import tensorflow as tf


# NOTE: As of now this implementation does not use Compression or Bottleneck.


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
