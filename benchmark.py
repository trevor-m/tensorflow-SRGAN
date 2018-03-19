import numpy as np
import glob
import os
from scipy import misc
from skimage.measure import compare_ssim
from skimage.color import rgb2ycbcr,rgb2yuv

from skimage.measure import compare_psnr
from utilities import preprocess, downsample

class Benchmark:
  """A collection of images to test a model on."""

  def __init__(self, path, name):
    self.path = path
    self.name = name
    #self.images_lr, self.names = self.load_images_by_model(model='LR')
    self.images_hr, self.names = self.load_images_by_model(model='HR')
    self.images_lr = []
    for img in self.images_hr:
      self.images_lr.append(downsample(img, 4))
    
  def load_images_by_model(self, model, file_format='png'):
    """Loads all images that match '*_{model}.{file_format}' and returns sorted list of filenames and names"""
    # Get files that match the pattern
    filenames = sorted(glob.glob(os.path.join(self.path, '*_' + model + '.' + file_format)))
    # Extract name/prefix eg: '/.../baby_LR.png' -> 'baby'
    names = [os.path.basename(x).split('_')[0] for x in filenames]
    return self.load_images(filenames), names

  def load_images(self, images):
    """Given a list of file names, return a list of images"""
    out = []
    for image in images:
      out.append(misc.imread(image, mode='RGB').astype(np.uint8))
    return out

  def deprocess(self, image):
    """Deprocess image output by model (from -1 to 1 float to 0 to 255 uint8)"""
    image = np.clip(255 * 0.5 * (image + 1.0), 0.0, 255.0).astype(np.uint8)
    return image

  def luminance(self, image):
    # Get luminance
    lum = rgb2ycbcr(image)[:,:,0]
    # Crop off 4 border pixels
    lum = lum[4:lum.shape[0]-4, 4:lum.shape[1]-4]
    #lum = lum.astype(np.float64)
    return lum

  def PSNR(self, gt, pred):
    #gt = gt.astype(np.float64)
    #pred = pred.astype(np.float64)
    #mse = np.mean((pred - gt)**2)
    #psnr = 10*np.log10(255*255/mse)
    #return psnr
    return compare_psnr(gt, pred, data_range=255)
    
  def SSIM(self, gt, pred):
    ssim = compare_ssim(gt, pred, data_range=255, gaussian_weights=True)
    return ssim
    
  def test_images(self, gt, pred):
    """Applies metrics to compare image lists pred vs gt"""
    avg_psnr = 0
    avg_ssim = 0
    individual_psnr = []
    individual_ssim = []

    for i in range(len(pred)):
      # compare to gt
      psnr = self.PSNR(self.luminance(gt[i]), self.luminance(pred[i]))
      ssim = self.SSIM(self.luminance(gt[i]), self.luminance(pred[i]))
      # save results to log_path ex: 'results/experiment1/Set5/baby/1000.png'
      #if save_images:
      #  path = os.path.join(log_path, self.name, self.names[i])
      # gather results
      individual_psnr.append(psnr)
      individual_ssim.append(ssim)
      avg_psnr += psnr
      avg_ssim += ssim
    
    avg_psnr /= len(pred)
    avg_ssim /= len(pred)
    return avg_psnr, avg_ssim, individual_psnr, individual_ssim
    
  def validate(self):
    """Tests metrics by using images output by other models"""
    for model in ['bicubic', 'SRGAN-MSE', 'SRGAN-VGG22', 'SRGAN-VGG54', 'SRResNet-MSE', 'SRResNet-VGG22']:
      model_output,_ = self.load_images_by_model(model)
      psnr, ssim, _, _ = self.test_images(self.images_hr, model_output)
      print('Validate %-6s for %-14s: PSNR: %.2f, SSIM: %.4f' % (self.name, model, psnr, ssim))

  def save_image(self, image, path):
    if not os.path.exists(os.path.split(path)[0]):
      os.makedirs(os.path.split(path)[0])
    misc.toimage(image, cmin=0, cmax=255).save(path)

  def save_images(self, images, log_path, iteration):
    count = 0
    for output, lr, hr, name in zip(images, self.images_lr, self.images_hr, self.names):
      # Save output
      path = os.path.join(log_path, self.name, name, '%d_out.png' % iteration)
      self.save_image(output, path)
      # Save ground truth
      path = os.path.join(log_path, self.name, name, '%d_hr.png' % iteration)
      self.save_image(hr, path)
      # Save low res
      path = os.path.join(log_path, self.name, name, '%d_lr.png' % iteration)
      self.save_image(lr, path)

      # Hack so that we only do first 14 images in BSD100 instead of the whole thing
      count += 1
      if count >= 14:
        break

  def evaluate(self, sess, g_y_pred, log_path=None, iteration=0):
    """Evaluate benchmark, returning the score and saving images."""
    pred = []
    for i, lr in enumerate(self.images_lr):
      # feed images 1 by 1 because they have different sizes
      lr = lr / 255.0
      output = sess.run(g_y_pred, feed_dict={'d_training:0': False, 'g_training:0': False, 'input_lowres:0': lr[np.newaxis]})
      # deprocess output
      pred.append(self.deprocess(np.squeeze(output, axis=0)))
    # save images
    if log_path:
      self.save_images(pred, log_path, iteration)
    return self.test_images(self.images_hr, pred)
