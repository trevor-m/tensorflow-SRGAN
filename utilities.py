import tensorflow as tf
import numpy as np
import pickle

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

def build_input_pipeline(filenames, batch_size, img_size, random_crop=False, shuffle=True, num_threads=1):
  """Builds a tensor which provides randomly sampled pictures from the list of filenames provided"""
  train_file_list = tf.constant(filenames)
  filename_queue = tf.train.string_input_producer(train_file_list, shuffle=shuffle)
  image = process_individual_image(filename_queue.dequeue(), img_size, random_crop)
  image_batch = tf.train.batch([image], batch_size=batch_size,
                                           num_threads=num_threads,
                                           capacity=10 * batch_size)
  return image_batch

def build_inputs(args):
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
      val_filenames = np.array(pickle.load(fo))[:args.test_size]
    with open('eval_indexes.pickle', 'rb') as fo:
      eval_indexes = np.array(pickle.load(fo))
    eval_filenames = train_filenames[eval_indexes[:args.test_size]]
  
  # Load first 5 val and eval files into memory (for test images)
  val_data = create_tensor_from_files(val_filenames[:5], sess, img_size=IMAGE_SIZE_TEST)
  eval_data = create_tensor_from_files(eval_filenames[:5], sess, img_size=IMAGE_SIZE_TEST)

  # Create input pipelines
  get_train_batch = build_input_pipeline(train_filenames, batch_size=args.batch_size, img_size=IMAGE_SIZE_TRAIN, random_crop=True)
  get_val_batch = build_input_pipeline(val_filenames, batch_size=args.batch_size, img_size=IMAGE_SIZE_TEST)
  get_eval_batch = build_input_pipeline(eval_filenames, batch_size=args.batch_size, img_size=IMAGE_SIZE_TEST)