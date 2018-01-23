import tensorflow as tf
import tensorflow.contrib.slim as slim

# VGG19 net
def vgg_19(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           reuse = False,
           fc_conv_padding='VALID'):
  """Oxford Net VGG 19-Layers version E Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      return net, end_points