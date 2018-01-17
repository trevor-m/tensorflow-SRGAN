import numpy as np
import glob
import os
from scipy import misc
from skimage.measure import compare_ssim
from utilities import preprocess

class Benchmark:
  """A collection of images to test a model on."""

  def __init__(self, path, name):
    self.name = name
    self.images_lr, self.images_hr, self.names = self.collect_images(path)

  def collect_images(self, path, file_format='png'):
    lr_names = glob.glob(os.path.join(path, '*_LR.'+file_format))
    lr = self.load_images(lr_names))
    hr = self.load_images(glob.glob(os.path.join(path, '*_HR.'+file_format)))
    # get name for each, eg: 'C:/.../baby_LR.png' -> 'baby'
    lr_names = [os.path.basename(x).split('_LR.')[0] for x in lr_names]
    return lr, hr, lr_names

  def load_images(self, images):
    out = []
    for image in images:
      out.append(misc.imread(image, mode='RGB').astype(np.float32))
    return out

  def deprocess(self, image, hr=False):
    """Deprocess image to 0,255 range"""
    if hr:
      return 255.0 * 0.5 * (image + 1.0)
    return 255.0 * image

  def luminance(self, image):
    return image[:,:,0] * 0.2126 + image[:,:,1] * 0.7152 + 0.0722 * image[:,:,2]

  def PSNR(self, gt, image):
    mse = np.mean((image - gt)**2)
    if mse == 0:
      return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))
    
  def SSIM(self, gt, image):
    return compare_ssim(gt, image)

  def evaluate(self, sess, g_y_pred, save_images=False, log_path=None, iteration=None):
    """Evaluate benchmark, returning the score and saving images."""
    avg_psnr = 0
    avg_ssim = 0
    individual_psnr = []
    individual_ssim = []

    for i, lr in enumerate(self.images_lr):
      # feed images 1 by 1 because they have different sizes
      output = sess.run(g_y_pred, feed_dict={'training:0': False, 'g_x:0': lr[np.newaxis]})
      # deprocess
      hr_pred = self.deprocess(np.squeeze(output, axis=0))
      # compare to gt
      psnr = self.PSNR(self.luminance(self.images_hr[i]), self.luminance(hr_pred))
      ssim = self.SSIM(self.luminance(self.images_hr[i]), self.luminance(hr_pred))

      # save results to log_path ex: 'results/experiment1/Set5/baby/1000.png'
      #if save_images:
      #  path = os.path.join(log_path, self.name, self.names[i])


      # gather results
      individual_psnr.append(psnr)
      individual_ssim.append(ssim)
      avg_psnr += psnr
      avg_ssim += ssim
    
    avg_psnr /= len(self.images_lr)
    avg_ssim /= len(self.images_lr)
    return avg_psnr, avg_ssim, individual_psnr, individual_ssim
