import tensorflow as tf
from vgg19 import vgg_19

class SRGanGenerator:
  """SRGAN Generator Model from Ledig et. al. 2017
  
  Reference: https://arxiv.org/pdf/1609.04802.pdf
  """
  def __init__(self, discriminator, training, content_loss='mse', use_gan=True, learning_rate=1e-4, num_blocks=16, num_upsamples=2):
    self.learning_rate = learning_rate
    self.num_blocks = num_blocks
    self.num_upsamples = num_upsamples
    self.use_gan = use_gan
    self.discriminator = discriminator
    self.training = training
    self.reuse_vgg = False
    if content_loss not in ['mse', 'L1', 'vgg22', 'vgg54']:
      print('Invalid content loss function. Must be \'mse\', \'vgg22\', or \'vgg54\'.')
      exit()
    self.content_loss = content_loss

  def ResidualBlock(self, x, kernel_size, filters, strides=1):
    """Residual block a la ResNet"""
    skip = x
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=self.training)
    x = tf.contrib.keras.layers.PReLU(shared_axes=[1,2])(x)
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=self.training)
    x = x + skip
    return x

  def Upsample2xBlock(self, x, kernel_size, filters, strides=1):
    """Upsample 2x via SubpixelConv"""
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
    x = tf.depth_to_space(x, 2)
    x = tf.contrib.keras.layers.PReLU(shared_axes=[1,2])(x)
    return x

  def forward(self, x):
    """Builds the forward pass network graph"""
    with tf.variable_scope('generator') as scope:
      x = tf.layers.conv2d(x, kernel_size=9, filters=64, strides=1, padding='same')
      x = tf.contrib.keras.layers.PReLU(shared_axes=[1,2])(x)
      skip = x

      # B x ResidualBlocks
      for i in range(self.num_blocks):
        x = self.ResidualBlock(x, kernel_size=3, filters=64, strides=1)

      x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same', use_bias=False)
      x = tf.layers.batch_normalization(x, training=self.training)
      x = x + skip

      # Upsample blocks
      for i in range(self.num_upsamples):
        x = self.Upsample2xBlock(x, kernel_size=3, filters=256)
      
      x = tf.layers.conv2d(x, kernel_size=9, filters=3, strides=1, padding='same', name='forward')
      return x
      
  def vgg_forward(self, x, layer, scope):
    # apply vgg preprocessing
    # move to range 0-255
    x = 255.0 * (0.5 * (x + 1.0))
    # subtract means
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean') # RGB means from VGG paper
    x = x - mean
    # convert to BGR
    x = x[:,:,:,::-1]
    # send through vgg19
    _,layers = vgg_19(x, is_training=False, reuse=self.reuse_vgg)
    self.reuse_vgg = True
    return layers[scope + layer]

  def _content_loss(self, y, y_pred):
    """MSE, VGG22, or VGG54"""
    if self.content_loss == 'mse':
      return tf.reduce_mean(tf.square(y - y_pred))
    if self.content_loss == 'L1':
      return tf.reduce_mean(tf.abs(y - y_pred))
    if self.content_loss == 'vgg22':
      with tf.name_scope('vgg19_1') as scope:
        vgg_y = self.vgg_forward(y, 'vgg_19/conv2/conv2_2', scope)
      with tf.name_scope('vgg19_2') as scope:
        vgg_y_pred = self.vgg_forward(y_pred, 'vgg_19/conv2/conv2_2', scope)
      return 0.006*tf.reduce_mean(tf.square(vgg_y - vgg_y_pred)) + 2e-8*tf.reduce_sum(tf.image.total_variation(y_pred))
      
    if self.content_loss == 'vgg54':
      with tf.name_scope('vgg19_1') as scope:
        vgg_y = self.vgg_forward(y, 'vgg_19/conv5/conv5_4', scope)
      with tf.name_scope('vgg19_2') as scope:
        vgg_y_pred = self.vgg_forward(y_pred, 'vgg_19/conv5/conv5_4', scope)
      return 0.006*tf.reduce_mean(tf.square(vgg_y - vgg_y_pred))

  def _adversarial_loss(self, y_pred):
    """For GAN."""
    y_discrim, y_discrim_logits = self.discriminator.forward(y_pred)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_discrim_logits, labels=tf.ones_like(y_discrim_logits)))

  def loss_function(self, y, y_pred):
    """Loss function"""
    if self.use_gan:
      # Weighted sum of content loss and adversarial loss
      return self._content_loss(y, y_pred) + 1e-3*self._adversarial_loss(y_pred)
    # Content loss only
    return self._content_loss(y, y_pred)
  
  def optimize(self, loss):
    #tf.control_dependencies([discrim_train
    # update_ops needs to be here for batch normalization to work
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
    with tf.control_dependencies(update_ops):
      return tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))


class SRGanDiscriminator:
  """SRGAN Discriminator Model from Ledig et. al. 2017
  
  Reference: https://arxiv.org/pdf/1609.04802.pdf
  """
  def __init__(self, training, learning_rate=1e-4, image_size=96):
    self.graph_created = False
    self.learning_rate = learning_rate
    self.training = training
    self.image_size = image_size

  def ConvolutionBlock(self, x, kernel_size, filters, strides):
    """Conv2D + BN + LeakyReLU"""
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=self.training)
    x = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x

  def forward(self, x):
    """Builds the forward pass network graph"""
    with tf.variable_scope('discriminator') as scope:
      # Reuse variables when graph is applied again
      if self.graph_created:
        scope.reuse_variables()
      self.graph_created = True

      # Image dimensions are fixed to the training size because of the FC layer
      x.set_shape([None, self.image_size, self.image_size, 3])

      x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same')
      x = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)(x)

      x = self.ConvolutionBlock(x, 3, 64, 2)
      x = self.ConvolutionBlock(x, 3, 128, 1)
      x = self.ConvolutionBlock(x, 3, 128, 2)
      x = self.ConvolutionBlock(x, 3, 256, 1)
      x = self.ConvolutionBlock(x, 3, 256, 2)
      x = self.ConvolutionBlock(x, 3, 512, 1)
      x = self.ConvolutionBlock(x, 3, 512, 2)

      x = tf.contrib.layers.flatten(x)
      x = tf.layers.dense(x, 1024)
      x = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)(x)
      logits = tf.layers.dense(x, 1)
      x = tf.sigmoid(logits)
      return x, logits

  def loss_function(self, y_real_pred, y_fake_pred, y_real_pred_logits, y_fake_pred_logits):
    """Discriminator wants to maximize log(y_real) + log(1-y_fake)."""
    loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(y_real_pred_logits), y_real_pred_logits))
    loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake_pred_logits), y_fake_pred_logits))
    return loss_real + loss_fake

  def optimize(self, loss):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
    with tf.control_dependencies(update_ops):
      return tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
