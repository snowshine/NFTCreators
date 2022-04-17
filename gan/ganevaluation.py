# ### IS and FID calculation
# Reference: 
# - machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
# - machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

import numpy as np
from numpy.random import shuffle
from math import floor
from scipy.linalg import sqrtm
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Sequential, Model, load_model

# from keras.datasets.mnist import load_data
from keras.datasets import cifar10

class ganevaluation:
	def __init__(self, collection, datapath, outputpath, preview_img_square = 3):
		# load the inception v3 model
		# model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
		self.model = InceptionV3()
		self.collection = collection
		self.batch_size = preview_img_square * preview_img_square
		self.preview_img_square = preview_img_square # preview generated samples in square, i.e 3x3=9
		self.data_path = datapath
		self.output_path = outputpath + collection
		self.generator_path = outputpath + collection + '/generator'

	# calculate inception score
	def calculate_inception_score(self, images, n_split=10, eps=1E-16):
		if images.shape[0] < n_split:
			n_split = images.shape[0]
		n_part = floor(images.shape[0] / n_split)

		# enumerate splits of images/predictions
		scores = list()
		for i in range(n_split):
			# retrieve images
			ix_start, ix_end = i * n_part, (i+1) * n_part
			subset = images[ix_start:ix_end]
			# predict p(y|x)
			p_yx = self.model.predict(subset)
			# calculate p(y)
			p_y = np.expand_dims(p_yx.mean(axis=0), 0)
			# calculate KL divergence using log probabilities
			kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
			# sum over classes
			sum_kl_d = kl_d.sum(axis=1)
			# average over images
			avg_kl_d = np.mean(sum_kl_d)
			# undo the log
			is_score = np.exp(avg_kl_d)
			# store
			scores.append(is_score)
		# average across images
		is_avg, is_std = np.mean(scores), np.std(scores)
		return is_avg, is_std
 
	# calculate frechet inception distance
	def calculate_fid(self, images1, images2):
		# calculate activations
		act1 = self.model.predict(images1)
		act2 = self.model.predict(images2)
		# calculate mean and covariance statistics
		mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
		mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
		# calculate sum squared difference between means
		ssdiff = np.sum((mu1 - mu2)**2.0)
		# calculate sqrt of product between cov
		covmean = sqrtm(sigma1.dot(sigma2))
		# check and correct imaginary numbers from sqrt
		if np.iscomplexobj(covmean):
			covmean = covmean.real
		# calculate score
		fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
		return fid

	def evaluate_epoch(self, epoch):
		# get both original and generated images
		images_orig, images_gen = self.get_images(epoch)
  		
		# calculate inception score for both original and generated images
		o_is_avg, o_is_std = self.calculate_inception_score(images_orig)
		g_is_avg, g_is_std = self.calculate_inception_score(images_gen)
  
  		# calculate fid
		fid = self.calculate_fid(images_gen, images_orig)#
		
		return o_is_avg, o_is_std, g_is_avg, g_is_std, fid
  
	def get_images(self, epoch):
		images_orig = self.load_origin_images()
		images_gen = self.load_generated_images(epoch)
		# print('loaded:', images_orig.shape, images_gen.shape)

		self.view_generated_images(images_gen)
 
		shuffle(images_orig)
		shuffle(images_gen)
 
		images_orig = self.prepare_inception_images(images_orig)
		images_gen = self.prepare_inception_images(images_gen)
		# print('prepared:', images_orig.shape, images_gen.shape)
		return images_orig, images_gen
	
	def load_generated_images(self, epoch):
		# use generator to generate images
		gen_path = self.generator_path + str(epoch)
		generator = load_model(gen_path, compile=False)
		generator.compile(optimizer='adam', loss='binary_crossentropy')

		SEED_DIM = 100 # Size vector to generate images from
		noise = tf.random.normal([self.batch_size, SEED_DIM])
		generated_image = generator(noise, training=False)#

		return generated_image.numpy()
	
	def load_origin_images(self):
		# load original images used for training
  		np_data = np.load(self.data_path + self.collection + '.npz')['arr_0'][:self.batch_size]
  		return np_data
	
	def scale_images(self, images, new_shape):
		# scale an array of images to a new size
		images_list = list()
		for image in images:
			# resize with nearest neighbor interpolation
			new_image = resize(image, new_shape, 0)
			# store
			images_list.append(new_image)
		return np.asarray(images_list)

	# assumes images have any shape and pixels in [0,255]
	def prepare_inception_images(self, images):
		# convert from uint8 to float32
		images = images.astype('float32')
		# scale images to the required size
		images = self.scale_images(images, (299,299,3))
		# pre-process images, scale to [-1,1]
		images = preprocess_input(images)
		return images
	
	def view_generated_images(self, imgs):
  		# scale generated image from [-1,1] to [0,1]
		imgs = (imgs + 1) / 2.0
		imgs = imgs.astype('float32')
			
		fig = plt.figure(figsize=(6, 6))
		# fig = plt.figure(figsize=(3, 3))
		n = self.preview_img_square
		for i in range(n * n):		
			plt.subplot(n, n, 1 + i)		
			plt.axis('off')
			# plot raw pixel data
			if (i + 1)%2 == 0:	
	 			plt.imshow(imgs[i, :, :, 0], cmap='gray')
			else:
	 			plt.imshow(imgs[i])
	
		plt.show()

	def loss_metrics_chart(self, whichchart):
		# load and show training loss metrics, either at batch or epoch level
		df = pd.read_pickle(self.output_path + '/train_loss.pkl')
  
		if whichchart == 'epoch': # group by epoch
			df = df.groupby('epoch').mean()
			xlabel = 'training run at epoch level'
			fsize = [12, 10]
		else:
			xlabel = 'training run at batch level'
			fsize = [18, 12]
  
		fig, ax = plt.subplots(figsize=fsize)	

		plt.plot(df.index, df["g_loss"], label='generator loss')
		plt.plot(df.index, df["d_loss"], label='discriminator loss')
		plt.plot(df.index, df["g_loss"]+df["d_loss"], label='total loss')

		ax.set_xlabel(xlabel)
		# ax.set_ylabel('loss')
  
		plt.title("Generator and Discriminator Loss During Training")
		plt.legend()
		plt.show()

	def get_prepared_cifar10(self):
		# calculate inception score for cifar-10 in Keras
		# load cifar10 images#
		(images_train, _), (images_test, _) = cifar10.load_data()
		shuffle(images_train)
		images_train = images_train[:self.batch_size]
		images_test = images_test[:self.batch_size]
		print('Loaded', images_train.shape, images_test.shape)
		
		images_train = self.prepare_inception_images(images_train)
		images_test = self.prepare_inception_images(images_test)
		print('prepared:', images_train.shape, images_test.shape)
		
		return images_train, images_test

	def epoch_evaluation(self, epochlist):
		# manual evaluation of generated images as well as IS and FID scores
		for epoch in epochlist:  
			o_is_avg, o_is_std, g_is_avg, g_is_std, fid = ganeval.evaluate_epoch(epoch)
			print('Collection:', collection, ' at epoch ', epoch)
			print('IS score for original images', o_is_avg, ' with std ', o_is_std)
			print('IS score for generated images', g_is_avg, ' with std ', g_is_std)
			print('FID: %.3f' % fid)
