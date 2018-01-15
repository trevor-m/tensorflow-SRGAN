import tensorflow as tf

class SRGanGenerator:
  """SRGAN Generator Model from Ledig et. al. 2017
  
  Reference: https://arxiv.org/pdf/1609.04802.pdf
  """
  def __init__(self, discriminator, learning_rate=1e-4, num_blocks=16, num_upsamples=2):
    self.learning_rate = learning_rate
    self.num_blocks = num_blocks
    self.num_upsamples = num_upsamples
    self.discriminator = discriminator
    #self.vgg

  def ResidualBlock(self, x, kernel_size, filters, strides=1):
    """Residual block a la ResNet"""
    skip = x
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
    x = tf.layers.batch_normalization(x)
    x = tf.contrib.keras.layers.PReLU(x)
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
    x = tf.layers.batch_normalization(x)
    x = x + skip
    return x

  def Upsample2xBlock(self, x, kernel_size, filters, strides=1):
    """Upsample 2x via SubpixelConv"""
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
    x = tf.depth_to_space(x, 2)
    x = tf.contrib.keras.layers.PReLU(x)
    return x

  def forward(self, x):
    """Builds the forward pass network graph"""
    with tf.variable_scope('generator') as scope:
      x = tf.layers.conv2d(x, kernel_size=9, filters=64, strides=1, padding='same')
      x = tf.contrib.keras.layers.PReLU(x)
      skip = x

      # B x ResidualBlocks
      for i in range(num_blocks):
        x = self.ResidualBlock(x, kernel_size=3, filters=64, strides=1)

      x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same')
      x = tf.layers.batch_normalization(x)
      x = x + skip

      # Upsample blocks
      for i in range(num_upsamples):
        x = self.Upsample2xBlock(x, kernel_size=3, filters=256)
      
      x = tf.layers.conv2d(x, kernel_size=9, filters=3, strides=1, padding='same', name='forward')
      return x

  def _content_loss(self, y, y_pred, name='content_loss'):
    """MSE"""
    # TODO VGG-based content loss
    return tf.reduce_mean(tf.square(y - y_pred), name=name)

  def _adversarial_loss(self, y_pred, name='adversarial_loss'):
    """GAN"""
    if not self.discriminator:
      print('Please supply a discriminator network when initializing model.')
      exit()
    
    y_discrim = self.discriminator.forward(y_pred, reuse=True)
    return tf.reduce_sum(-tf.log(y_discrim), name=name)

  def _perceptual_loss(self, y, y_pred, name='loss'):
    """Weighted sum of content and adversarial loss"""
    return tf.add(self._content_loss(y, y_pred), 1e-3*self._adversarial_loss(y), name=name)


  def loss_function(self, y_pred, y):
    """Loss function"""
    return self._perceptual_loss(y, y_pred, name='loss')
  
  def optimize(self, loss):
    # TODO limit variables trained to only generator
    return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)


class SRGanDiscriminator:
  """SRGAN Discriminator Model from Ledig et. al. 2017
  
  Reference: https://arxiv.org/pdf/1609.04802.pdf
  """

  def ConvolutionBlock(self, x, kernel_size, filters, strides):
    """Conv2D + BN + LeakyReLU"""
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
    x = tf.layers.batch_normalization(x)
    x = tf.contrib.keras.layers.LeakyReLU(x)
    return x

  def forward(self, x, reuse=False):
    """Builds the forward pass network graph"""
    with tf.variable_scope('discriminator') as scope:
      if reuse:
        scope.reuse_variables()

      x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same')
      x = tf.contrib.keras.layers.LeakyReLU(x, alpha=0.2)

      x = self.ConvolutionBlock(x, 3, 64, 2)
      x = self.ConvolutionBlock(x, 3, 128, 1)
      x = self.ConvolutionBlock(x, 3, 128, 2)
      x = self.ConvolutionBlock(x, 3, 256, 1)
      x = self.ConvolutionBlock(x, 3, 256, 2)
      x = self.ConvolutionBlock(x, 3, 512, 1)
      x = self.ConvolutionBlock(x, 3, 512, 2)

      x = tf.layers.dense(x, 1024)
      x = tf.contrib.keras.layers.LeakyReLU(x, aplha=0.2)
      x = tf.layers.dense(x, 1)
      x = tf.sigmoid(x, name='forward')
      return x

  def loss_function(self, y, y_pred):
    return 0

  def optimize(self, loss):
    # TODO limit variables trained
    return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
