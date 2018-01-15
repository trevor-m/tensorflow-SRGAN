import tensorflow as tf
from tensorflow.python.training import queue_runner
import numpy as np
import argparse
import srgan
import os
from utilities import build_input_pipeline

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--load', type=str, help='Checkpoint to load all weights from.')
  parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for Adam.')
  parser.add_argument('--image-size-train', type=int, default=352, help='Dimensions of training images.')
  parser.add_argument('--image-size-test', type=int, default=512, help='Dimensions of testing images.')
  parser.add_argument('--num-test', type=int, default=1000, help='Number of images to test on.')
  args = parser.parse_args()
  
  # Set up models
  discriminator = srgan.SRGanDiscriminator()
  generator = srgan.SRGanGenerator(discriminator=discriminator, learning_rate=args.learning_rate)
  # Discriminator
  d_x = tf.placeholder(tf.float32, [None, None, None, 3])
  d_y = tf.placeholder(tf.float32, [None, 1])
  d_y_pred = discriminator.forward(d_x)
  d_loss = discriminator.loss_function(d_y, d_y_pred)
  d_train_step = discriminator.optimize(d_loss)
  # Generator
  g_x = tf.placeholder(tf.float32, [None, None, None, 3])
  g_y = tf.placeholder(tf.float32, [None, None, None, 3])
  g_y_pred = generator.forward(g_x)
  g_loss = generator.loss_function(g_y, g_y_pred)
  g_train_step = generator.optimize(g_loss)

  # Input Pipeline
  # TODO overfit
  # TODO args.num_test
  # TODO image examples
  get_batch = build_input_pipeline('train.pickle', args.batch_size, args.image_size_train, random_crop=True)
  get_batch_val = build_input_pipeline('val.pickle', args.batch_size, args.image_size_test)
  get_batch_eval = build_input_pipeline('eval.pickle', args.batch_size, args.image_size_test)

  # TODO create log folder
  log_path = 'test/'

  # Train
  with tf.Session() as sess:
    # Initialize
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # Start input pipeline thread(s)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # Load saved weights
    iteration = 0
    saver = tf.train.Saver()
    if args.load:
      iteration = int(args.load.split('-')[-1])
      saver.restore(sess, args.load)

    while True:
      if iteration % args.log_freq == 0:
        # TODO Test
        # Save checkpoint
        saver.save(sess, log_path, global_step=iteration, write_meta_graph=False)

      # Get data
      feed_dict = {} #TODO

      # Train discriminator
      sess.run(d_train_step, feed_dict=feed_dict)
      # Train generator
      sess.run(g_train_step, feed_dict=feed_dict)

      iteration += 1

    # Stop queue threads
    coord.request_stop()
    coord.join(threads)

  
if __name__ == "__main__":
  main()