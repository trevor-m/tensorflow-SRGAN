import tensorflow as tf
import numpy as np
import pickle
import skimage.transform
import skimage.filters
import datetime
import os
import shutil
import math
from scipy import misc
import scipy.ndimage

def process_individual_image(filename_queue, img_size, random_crop=False):
  """Individual loading & processing for each image"""
  image_file = tf.read_file(filename_queue)
  image = tf.image.decode_image(image_file, 3)
  if random_crop:
    # for training, take a random crop of the image
    image_shape = tf.shape(image)
    # if smaller than img_size, pad with 0s to prevent error
    image = tf.image.pad_to_bounding_box(image, 0, 0, tf.maximum(img_size, image_shape[0]), tf.maximum(img_size, image_shape[1]))
    image = tf.random_crop(image, size=[img_size, img_size, 3])
    image.set_shape((img_size, img_size, 3))
  else:
    # for testing or when dealing with encodings, always take a center crop of the image
    image = tf.image.resize_image_with_crop_or_pad(image, img_size, img_size)
    image.set_shape((img_size, img_size, 3))
  image = (tf.cast(image, dtype=tf.float32) / tf.constant(255.0))# * 2.0 - 1.0
  return image

def create_tensor_from_files(files, sess, img_size):
  """Given a list of image files, returns a np.array ready to be used as a test batch"""
  # load images
  images = np.zeros((len(files), img_size, img_size, 3))
  for i, image_file in enumerate(files):
    images[i,:,:,:] = sess.run(process_individual_image(tf.constant(files[i]), img_size, random_crop=False))
  return images

def build_input_pipeline(filenames, batch_size, img_size, random_crop=False, shuffle=True, num_threads=1):
  """Builds a tensor which provides randomly sampled pictures from the list of filenames provided"""
  train_file_list = tf.constant(filenames)
  filename_queue = tf.train.string_input_producer(train_file_list, shuffle=shuffle)
  image = process_individual_image(filename_queue.dequeue(), img_size, random_crop)
  image_batch = tf.train.batch([image], batch_size=batch_size,
                                           num_threads=num_threads,
                                           capacity=10 * batch_size)
  return image_batch

def build_inputs(args, sess):
  if args.overfit:
    # Overfit to a single image
    train_filenames = np.array(['overfit.png'])
    val_filenames = np.array(['overfit.png'])
    eval_filenames = np.array(['overfit.png'])
    #args.batch_size = 1
    args.num_test = 1
  else:
    # Regular dataset
    with open('train.pickle', 'rb') as fo:
      train_filenames = np.array(pickle.load(fo))
    with open('val.pickle', 'rb') as fo:
      val_filenames = np.array(pickle.load(fo))[:args.num_test]
    with open('eval_indexes.pickle', 'rb') as fo:
      eval_indexes = np.array(pickle.load(fo))
    eval_filenames = train_filenames[eval_indexes[:args.num_test]]
  
  # Load first 5 val and eval files into memory (for test images)
  val_data = create_tensor_from_files(val_filenames[:5], sess, img_size=args.image_size_test)
  eval_data = create_tensor_from_files(eval_filenames[:5], sess, img_size=args.image_size_test)

  # Create input pipelines
  get_train_batch = build_input_pipeline(train_filenames, batch_size=args.batch_size, img_size=args.image_size_train, random_crop=True)
  get_val_batch = build_input_pipeline(val_filenames, batch_size=args.batch_size, img_size=args.image_size_train)
  get_eval_batch = build_input_pipeline(eval_filenames, batch_size=args.batch_size, img_size=args.image_size_train)
  return get_train_batch, get_val_batch, get_eval_batch, val_data, eval_data

def downsample(image, factor):
  """Downsampling function which matches photoshop"""
  sigma = (factor - 1.0) / 2
  image = skimage.filters.gaussian(image, sigma, multichannel=True, preserve_range=True)
  return skimage.transform.resize(image, (image.shape[0]//factor, image.shape[1]//factor, 3), order=1, preserve_range=True, mode='constant')

def downsample_batch(batch, factor):
  downsampled = np.zeros((batch.shape[0], batch.shape[1]//factor, batch.shape[2]//factor, 3))
  for i in range(batch.shape[0]):
    downsampled[i,:,:,:] = downsample(batch[i,:,:,:], factor)
  return downsampled

def build_log_dir(args, arguments):
  """Set up a timestamped directory for results and logs for this training session"""
  if args.name:
    log_path = args.name #(name + '_') if name else ''
  else:
    log_path = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  log_path = os.path.join('results', log_path)
  if not os.path.exists(log_path):
    os.makedirs(log_path)
  print('Logging results for this session in folder "%s".' % log_path)
  # Output csv header
  with open(log_path + '/loss.csv', 'a') as f:
    f.write('iteration, val_error, eval_error, set5_psnr, set5_ssim, set14_psnr, set14_ssim, bsd100_psnr, bsd100_ssim\n')
  # Copy this code to folder
  shutil.copy2('srgan.py', os.path.join(log_path, 'srgan.py'))
  shutil.copy2('train.py', os.path.join(log_path, 'train.py'))
  shutil.copy2('utilities.py', os.path.join(log_path, 'utilities.py'))
  # Write command line arguments to file
  with open(log_path + '/args.txt', 'w+') as f:
    f.write(' '.join(arguments))
  # Make directory for each visual example
  """num = 5
  if args.overfit:
    num = 1
  for i in range(num):
    full_path = os.path.join(log_path,'eval_'+str(i))
    if not os.path.exists(full_path):
      os.makedirs(full_path)
    full_path = os.path.join(log_path, 'val_'+str(i))
    if not os.path.exists(full_path):
      os.makedirs(full_path)
  """
  return log_path

def preprocess(lr, hr):
  """Preprocess lr and hr batch"""
  #lr = lr
  hr = hr * 2.0 - 1.0
  return lr, hr

def save_image(path, data, highres=False):
  # transform from [-1, 1] to [0, 1]
  if highres:
    data = (data + 1.0) * 0.5
  # transform from [0, 1] to [0, 255], clip, and convert to uint8
  data = np.clip(data * 255.0, 0.0, 255.0).astype(np.uint8)
  misc.toimage(data, cmin=0, cmax=255).save(path)

def evaluate_model(loss_function, get_batch, sess, num_images, batch_size):
  """Tests the model over all num_images using input tensor get_batch"""
  loss = 0
  total = 0
  for i in range(int(math.ceil(num_images/batch_size))):
    batch_hr = sess.run(get_batch)
    batch_lr = downsample_batch(batch_hr, factor=4)
    batch_lr, batch_hr = preprocess(batch_lr, batch_hr)
    loss += sess.run(loss_function, feed_dict={'g_training:0': False, 'd_training:0': False, 'input_lowres:0': batch_lr, 'input_highres:0':batch_hr})
    total += 1
  loss = loss / total
  return loss

def test_examples(sess, batch_hr, g_y_pred, epoch, log_path, prefix, num=5):
  """Test the first num(5) images and save the outputs"""
  # make feed dict with downsampled images
  batch_lr = downsample_batch(batch_hr, factor=4)
  batch_lr, batch_hr = preprocess(batch_lr, batch_hr)

  # run everything through network
  output = sess.run(g_y_pred, feed_dict={'g_training:0': False, 'd_training:0': False, 'input_lowres:0': batch_lr})
  
  # iterate over all 5 examples
  for i in range(output.shape[0]):
    # save full resolution input/output
    file_name = os.path.join(log_path, '%s_%d' % (prefix, i), '%d_in.png' % epoch)
    save_image(file_name, batch_hr[i, :, :, :], highres=True)
    file_name = os.path.join(log_path, '%s_%d' % (prefix, i), '%d_out.png' % epoch)
    save_image(file_name, output[i, :, :, :], highres=True)
    # save downsampled
    file_name = os.path.join(log_path, '%s_%d' % (prefix, i), '%d_down.png' % epoch)
    save_image(file_name, batch_lr[i, :, :, :])
