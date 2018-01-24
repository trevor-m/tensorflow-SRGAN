# SRGan in Tensorflow

This is an implementation of the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) using TensorFlow.

This work is currently in progress.

## Prequisites

* TensorFlow
* Python 3
* NumPy

## Usage

### Set up

1. Download the VGG19 weights provided by [TensorFlow-Slim](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz). Place the vgg_19.ckpt file in this directory.

2. Download the [benchmark images](https://twitter.box.com/s/lcue6vlrd01ljkdtdkhmfvk7vtjhetog) provided by the authors of the SRGAN paper. Extract the folder to this directory and rename it to "Benchmarks".

3. Run the `create_lists.py` script to make a list of filenames for your training data (TODO) OR download the pretrained weights (TODO).

## Results

TODO
