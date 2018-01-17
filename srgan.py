import tensorflow as tf

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
    if content_loss not in ['mse', 'vgg22', 'vgg54']:
      print('Invalid content loss function. Must be \'mse\', \'vgg22\', or \'vgg54\'.')
      exit()
    self.content_loss = content_loss

  def ResidualBlock(self, x, kernel_size, filters, strides=1):
    """Residual block a la ResNet"""
    skip = x
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
    x = tf.layers.batch_normalization(x, training=self.training)
    x = tf.contrib.keras.layers.PReLU(shared_axes=[1,2])(x)
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
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
      tf.contrib.keras.layers.PReLU(shared_axes=[1,2])(x)
      skip = x

      # B x ResidualBlocks
      for i in range(self.num_blocks):
        x = self.ResidualBlock(x, kernel_size=3, filters=64, strides=1)

      x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same')
      x = tf.layers.batch_normalization(x, training=self.training)
      x = x + skip

      # Upsample blocks
      for i in range(self.num_upsamples):
        x = self.Upsample2xBlock(x, kernel_size=3, filters=256)
      
      x = tf.layers.conv2d(x, kernel_size=9, filters=3, strides=1, padding='same', name='forward')
      return x

  def _content_loss(self, y, y_pred):
    """MSE, VGG22, or VGG54"""
    if self.content_loss == 'mse':
      return tf.reduce_mean(tf.square(y - y_pred), name='content_loss')
    if self.content_loss == 'vgg22':
      # TODO
      return 0 # * 1.0/12.75
    if self.content_loss == 'vgg54':
      # TODO
      return 0 # * 1.0/12.75

  def _adversarial_loss(self, y_pred):
    """GAN"""
    y_discrim = self.discriminator.forward(y_pred, reuse=False)
    return tf.reduce_sum(-tf.log(y_discrim), name='adversarial_loss')

  def _perceptual_loss(self, y, y_pred):
    """Weighted sum of content and adversarial loss"""
    return tf.add(self._content_loss(y, y_pred), 1e-3*self._adversarial_loss(y), name='loss')

  def loss_function(self, y, y_pred):
    """Loss function"""
    if self.use_gan:
      return self._perceptual_loss(y, y_pred, name='loss')
    return self._content_loss(y, y_pred)
  
  def optimize(self, loss):
    # TODO limit variables trained to only generator
    # update need to be added for batch normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)


class SRGanDiscriminator:
  """SRGAN Discriminator Model from Ledig et. al. 2017
  
  Reference: https://arxiv.org/pdf/1609.04802.pdf
  """
  def __init__(self, training, learning_rate=1e-4):
    self.graph_created = False
    self.learning_rate = learning_rate
    self.training = training

  def ConvolutionBlock(self, x, kernel_size, filters, strides):
    """Conv2D + BN + LeakyReLU"""
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
    x = tf.layers.batch_normalization(x, training=self.training)
    x = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x

  def forward(self, x):
    """Builds the forward pass network graph"""
    with tf.variable_scope('discriminator') as scope:
      if self.graph_created:
        scope.reuse_variables()
      self.graph_created = True

      x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same')
      x = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)(x)

      x = self.ConvolutionBlock(x, 3, 64, 2)
      x = self.ConvolutionBlock(x, 3, 128, 1)
      x = self.ConvolutionBlock(x, 3, 128, 2)
      x = self.ConvolutionBlock(x, 3, 256, 1)
      x = self.ConvolutionBlock(x, 3, 256, 2)
      x = self.ConvolutionBlock(x, 3, 512, 1)
      x = self.ConvolutionBlock(x, 3, 512, 2)

      x = tf.layers.dense(x, 1024)
      x = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)(x)
      x = tf.layers.dense(x, 1)
      x = tf.sigmoid(x, name='forward')
      return x

  def loss_function(self, y_real_pred, y_fake_pred):
    loss_real = tf.log(y_real_pred)
    loss_fake = tf.log(1-y_fake_pred)
    # TODO: alpha
    return tf.reduce_mean(loss_real + loss_fake)

  def optimize(self, loss):
    # TODO limit variables trained
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
