"""Training program for SRGAN implementation.

Benchmark data sets provided by the paper are available here: https://twitter.app.box.com/s/lcue6vlrd01ljkdtdkhmfvk7vtjhetog
"""
import tensorflow as tf
from tensorflow.python.training import queue_runner
import numpy as np
import argparse
import srgan
from benchmark import Benchmark
import os
import sys
from utilities import build_inputs, downsample_batch, build_log_dir, preprocess, evaluate_model, test_examples

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--load', type=str, help='Checkpoint to load all weights from.')
  parser.add_argument('--load-gen', type=str, help='Checkpoint to load generator weights only from.')
  parser.add_argument('--name', type=str, help='Name of experiment.')
  parser.add_argument('--overfit', action='store_true', help='Overfit to a single image.')
  parser.add_argument('--batch-size', type=int, default=16, help='Mini-batch size.')
  parser.add_argument('--log-freq', type=int, default=10000, help='How many training iterations between validation/checkpoints.')
  parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for Adam.')
  parser.add_argument('--content-loss', type=str, default='mse', choices=['mse', 'vgg22', 'vgg54'], help='Metric to use for content loss.')
  parser.add_argument('--use-gan', action='store_true', help='Add adversarial loss term to generator and trains discriminator.')
  parser.add_argument('--image-size', type=int, default=96, help='Size of random crops used for training samples.')
  parser.add_argument('--vgg-weights', type=str, default='vgg_19.ckpt', help='File containing VGG19 weights (tf.slim)')
  parser.add_argument('--train-dir', type=str, help='Directory containing training images')
  parser.add_argument('--validate-benchmarks', action='store_true', help='If set, validates that the benchmarking metrics are correct for the images provided by the authors of the SRGAN paper.')
  parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use')
  args = parser.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  
  # Set up models
  d_training = tf.placeholder(tf.bool, name='d_training')
  g_training = tf.placeholder(tf.bool, name='g_training')
  discriminator = srgan.SRGanDiscriminator(training=g_training, image_size=args.image_size)
  generator = srgan.SRGanGenerator(discriminator=discriminator, training=d_training, learning_rate=args.learning_rate, content_loss=args.content_loss, use_gan=args.use_gan)
  # Generator
  g_x = tf.placeholder(tf.float32, [None, None, None, 3], name='input_lowres')
  g_y = tf.placeholder(tf.float32, [None, None, None, 3], name='input_highres')
  g_y_pred = generator.forward(g_x)
  g_loss = generator.loss_function(g_y, g_y_pred)
  g_train_step = generator.optimize(g_loss)
  # Discriminator
  d_x_real = tf.placeholder(tf.float32, [None, None, None, 3], name='input_real')
  d_y_real_pred, d_y_real_pred_logits = discriminator.forward(d_x_real)
  d_y_fake_pred, d_y_fake_pred_logits = discriminator.forward(g_y_pred)
  d_loss = discriminator.loss_function(d_y_real_pred, d_y_fake_pred, d_y_real_pred_logits, d_y_fake_pred_logits)
  d_train_step = discriminator.optimize(d_loss)
  
  # Set up benchmarks
  benchmarks = [Benchmark('Benchmarks/Set5', name='Set5'),
                Benchmark('Benchmarks/Set14', name='Set14'),
                Benchmark('Benchmarks/BSD100', name='BSD100')]
  if args.validate_benchmarks:
    for benchmark in benchmarks:
      benchmark.validate()

  # Create log folder
  if args.load and not args.name:
    log_path = os.path.dirname(args.load)
  else:
    log_path = build_log_dir(args, sys.argv)

  with tf.Session() as sess:
    # Build input pipeline
    get_train_batch, get_val_batch, get_eval_batch = build_inputs(args, sess)
    # Initialize
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # Start input pipeline thread(s)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    # Load saved weights
    iteration = 0
    saver = tf.train.Saver()
    # Load generator
    if args.load_gen:
      gen_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))
      iteration = int(args.load_gen.split('-')[-1])
      gen_saver.restore(sess, args.load_gen)
    # Load all
    if args.load:
      iteration = int(args.load.split('-')[-1])
      saver.restore(sess, args.load)
    # Load VGG
    if 'vgg' in args.content_loss:
      vgg_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19'))
      vgg_saver.restore(sess, args.vgg_weights)

    # Train
    while True:
      if iteration % args.log_freq == 0:
        # Test every log-freq iterations
        val_error = evaluate_model(g_loss, get_val_batch, sess, 119, args.batch_size)
        eval_error = evaluate_model(g_loss, get_eval_batch, sess, 119, args.batch_size)
        # Log error
        print('[%d] Test: %.7f, Train: %.7f' % (iteration, val_error, eval_error), end='')
        # Evaluate benchmarks
        log_line = ''
        for benchmark in benchmarks:
          psnr, ssim, _, _ = benchmark.evaluate(sess, g_y_pred, log_path, iteration)
          print(' [%s] PSNR: %.2f, SSIM: %.4f' %( benchmark.name, psnr, ssim), end='')
          log_line += ',%.7f, %.7f' %(psnr, ssim)
        print()
        # Write to log
        with open(log_path + '/loss.csv', 'a') as f:
          f.write('%d, %.15f, %.15f%s\n' % (iteration, val_error, eval_error, log_line))
        # Save checkpoint
        saver.save(sess, os.path.join(log_path, 'weights'), global_step=iteration, write_meta_graph=False)

      # Train discriminator
      if args.use_gan:
        batch_hr = sess.run(get_train_batch)
        batch_lr = downsample_batch(batch_hr, factor=4)
        batch_lr, batch_hr = preprocess(batch_lr, batch_hr)
        sess.run(d_train_step, feed_dict={d_training: True, g_training: True, g_x: batch_lr, g_y: batch_hr, d_x_real: batch_hr})
      # Train generator
      batch_hr = sess.run(get_train_batch)
      batch_lr = downsample_batch(batch_hr, factor=4)
      batch_lr, batch_hr = preprocess(batch_lr, batch_hr)
      sess.run(g_train_step, feed_dict={d_training: True, g_training: True, g_x: batch_lr, g_y: batch_hr})

      iteration += 1

    # Stop queue threads
    coord.request_stop()
    coord.join(threads)

  
if __name__ == "__main__":
  main()
