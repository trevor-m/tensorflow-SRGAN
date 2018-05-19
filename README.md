# SRGan in Tensorflow

This is an implementation of the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) using TensorFlow.

## Usage

### Set up

1. Download the VGG19 weights provided by [TensorFlow-Slim](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz). Place the vgg_19.ckpt file in this directory.
2. Download a dataset of images. I recommend [ImageNet](http://image-net.org/challenges/LSVRC/2014/index) or [Places205](http://places.csail.mit.edu/index.html). Specify the directory containing your dataset using the `--train-dir` argument when training the model.

### Training

SRResNet-MSE
```
python train.py --name srresnet-mse --content-loss mse --train-dir path/to/dataset
```

SRResNet-VGG22
```
python train.py --name srresnet-vgg22 --content-loss vgg22 --train-dir path/to/dataset
```

SRGAN-MSE
```
python train.py --name srgan-mse --use-gan --content-loss mse --train-dir path/to/dataset --load results/srresnet-mse/weights-1000000
```

SRGAN-VGG22
```
python train.py --name srgan-vgg22 --use-gan --content-loss vgg22 --train-dir path/to/dataset --load results/srresnet-mse/weights-1000000
```

SRGAN-VGG54
```
python train.py --name srgan-vgg54 --use-gan --content-loss vgg54 --train-dir path/to/dataset --load results/srresnet-mse/weights-1000000
```

## Results
| **Set5** | Ledig SRResNet | This SRResNet | Ledig SRGAN | This SRGAN |
| --- | --- | --- | --- | --- |
| PSNR | 32.05 | 32.11 | 29.40 | 28.21 |
| SSIM | 0.9019| 0.8933 | 0.8472 | 0.8200 |

| **Set14** | Ledig SRResNet | This SRResNet | Ledig SRGAN | This SRGAN |
| --- | --- | --- | --- | --- |
| PSNR | 28.49 | 28.61 | 26.02 | 25.74 |
| SSIM | 0.8184| 0.7809 | 0.7397 | 0.6909 |

| **BSD100** | Ledig SRResNet | This SRResNet | Ledig SRGAN | This SRGAN |
| --- | --- | --- | --- | --- |
| PSNR | 27.58 | 27.57 | 25.16 | 24.80 |
| SSIM | 0.7620 | 0.7346 | 0.6688 | 0.6314 |
