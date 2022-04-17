# ### Imports and setup

import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, Rescaling
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import image_dataset_from_directory

import numpy as np
from numpy.random import random
from numpy.random import choice
import pandas as pd
import os 
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

# ### Build GAN models
# hyperparameter categories:
# - Number of Hidden Layers and Neuron Counts
# - Activation Functions
# - Advanced Activation Functions
# - Regularization: L1, L2, Dropout
# - Batch Normalization
# - Training Parameters

class dcgan:
    def __init__(self, collection, outputpath, startepoch, image_shape = (128,128,3)):        
        self.seed_dim = 100
        self.image_shape = image_shape
        # a random Gaussian weight initializer: zero-centered with a standard deviation of 0.02
        # adding gaussian noise to every layer of generator (Zhao et. al. EBGAN)
        self.init = RandomNormal(mean=0.0, stddev=0.02)
        self.kernelsize = 4
        self.dropout_rate = 0.25   #0.3; 0.4
        # Two Time-Scale Update Rule (TTUR): different learning rates for generator/discriminator         
        if image_shape[0] == 128:        
            self.g_lr = 1e-4
            self.d_lr = 4e-4
        else: #32
            self.g_lr = 1.5e-4
            self.d_lr = 2e-4  # 4e-4

        # Define Loss Function
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()
        # Optimizer: Adam        
        self.generator_optimizer = Adam(learning_rate=self.g_lr, beta_1 = 0.5)
        self.discriminator_optimizer = Adam(learning_rate=self.d_lr, beta_1 = 0.5)

        preview_examples_num = 16 # 4x4 square
        # You will reuse this seed overtime (so it's easier) to visualize progress
        self.PREVIEW_SEED = tf.random.normal([preview_examples_num, self.seed_dim])

        self.collection = collection
        self.output_path = outputpath + collection + '/'        

        generator, discriminator = self.get_models(startepoch)
        self.GENERATOR = generator
        self.DISCRIMINATOR = discriminator
        
    # #### Discriminator and Generator Model for 32x32x3 images
    # Reference: # https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb    
    def build_generator_32(self):
        model = Sequential(name='Generator32x32')
        
        # model.add(Dense(4*4*256, activation="relu", use_bias=False, input_dim=seed_dim))
        model.add(Dense(4*4*256, use_bias=False, input_dim=self.seed_dim))
        model.add(Reshape((4, 4, 256)))
        assert model.output_shape == (None, 4, 4, 256)  # Note: None is the batch size
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Conv2DTranspose(256, (3, 3), strides=(1, 1), use_bias=False, padding='same'))
        assert model.output_shape == (None, 4, 4, 256)
        model.add(BatchNormalization())
        model.add(LeakyReLU())    
        
        model.add(Conv2DTranspose(256, (3, 3), strides=(2, 2), use_bias=False, padding='same'))
        assert model.output_shape == (None, 8, 8, 256)
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), use_bias=False, padding='same'))
        assert model.output_shape == (None, 16, 16, 128)
        model.add(BatchNormalization())
        model.add(LeakyReLU())
    
        model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), use_bias=False, padding='same'))
        assert model.output_shape == (None, 32, 32, 64)
        model.add(BatchNormalization())
        model.add(LeakyReLU())    
    
        model.add(Conv2DTranspose(3, (3, 3), strides=(1, 1), use_bias=False, padding='same', activation='tanh'))
        assert model.output_shape == (None, 32, 32, 3)
    
        return model
    
    def build_discriminator_32(self, image_shape):
        model = Sequential(name='Discriminator32x32')
        
        # image_shape = (32,32,3)
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=image_shape))
        assert model.output_shape == (None, 16, 16, 32)
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(self.dropout_rate))    
    
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 8, 8, 64)
        # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(self.dropout_rate))
    
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 4, 4, 128)
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(self.dropout_rate))    
    
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
        assert model.output_shape == (None, 4, 4, 256)
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(self.dropout_rate))
    
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        assert model.output_shape == (None, 1)  
    
        return model    
    
    # #### Discriminator and Generator Model for 128x128x3 images    
    # Inputs: Point in latent space, e.g. a 100-element vector of Gaussian random numbers.
    # Outputs: 128x128x3 color image (3 channels) with pixel values in [-1,1]
    def build_generator(self):
        model = Sequential(name='Generator128x128')
        init = self.init 
        ksize = self.kernelsize
    
        # start with 256 tiny 4x4 images
        model.add(Dense(4*4*256, input_dim=self.seed_dim))
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(LeakyReLU(alpha=0.2))      
        model.add(Reshape((4,4,256)))
        assert model.output_shape == (None, 4, 4, 256) # None is the batch size
        
        # upsample to 8x8
        model.add(Conv2DTranspose(256, strides=2, kernel_initializer=init, kernel_size=ksize, padding='same')) 
        assert model.output_shape == (None, 8, 8, 256)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(256, strides=2, kernel_initializer=init, kernel_size=ksize, padding='same'))     
        assert model.output_shape == (None, 16, 16, 256)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
       
        model.add(Conv2DTranspose(128, strides=2, kernel_initializer=init, kernel_size=ksize, padding='same'))     
        assert model.output_shape == (None, 32, 32, 128) 
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))     
        
        model.add(Conv2DTranspose(128, strides=2, kernel_initializer=init, kernel_size=ksize, padding='same'))
        assert model.output_shape == (None, 64, 64, 128) 
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))    
        
        model.add(Conv2DTranspose(64, strides=2, kernel_initializer=init, kernel_size=ksize, padding='same'))
        assert model.output_shape == (None, 128, 128, 64)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))    
    
        # Final CNN layer
        color_channel = 3 # assume color
        model.add(Conv2D(color_channel, kernel_initializer=init, kernel_size=ksize, padding="same"))
        model.add(Activation("tanh"))
        assert model.output_shape == (None, 128, 128, 3)
    
        return model
    
    # The discriminator is a CNN-based image classifier
    # Inputs: Image shape=(height, width, depth) for 128x128x3 RGB pictures in data_format="channels_last"
    # Outputs: Binary classification, likelihood the sample is real (or fake).
    def build_discriminator(self, image_shape):        
        model = Sequential(name='Discriminator128x128')
        # init = self.init 
        ksize = self.kernelsize
        droprate = self.dropout_rate
    
        # start with 64 filters and input image size 128x128x3
        model.add(Conv2D(64, kernel_size=ksize, input_shape=image_shape, padding="same"))
        assert model.output_shape == (None, 128, 128, 64)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(droprate))
    
        # downsample to 64x64x3
        model.add(Conv2D(64, kernel_size=ksize, strides=2, padding="same"))
        assert model.output_shape == (None, 64, 64, 64)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(droprate))
    
        model.add(Conv2D(128, kernel_size=ksize, strides=2, padding="same"))
        assert model.output_shape == (None, 32, 32, 128)
        # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(droprate))    
    
        model.add(Conv2D(128, kernel_size=ksize, strides=2, padding="same"))
        assert model.output_shape == (None, 16, 16, 128)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(droprate))
    
        model.add(Conv2D(256, kernel_size=ksize, strides=2, padding="same"))
        assert model.output_shape == (None, 8, 8, 256)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(droprate))
    
        model.add(Conv2D(512, kernel_size=ksize, strides=2, padding="same"))
        assert model.output_shape == (None, 4, 4, 512)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(droprate))
    
        # classify    
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        assert model.output_shape == (None, 1)
     
        return model
            
    def smooth_positive_labels(self, y):
        # smoothing class=1 to [0.8, 1.2] assume y[-1,1]
        result = y - (random(y.shape) * 0.2)
        return result
    
    def discriminator_loss(self, real_output, fake_output):
        # loss when predict a real image is real
        # real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        # one-sided Label Smoothing: tune down discriminator to avoid overconfidence.
        real_loss = self.cross_entropy(self.smooth_positive_labels(tf.ones_like(real_output)), real_output)
        # real_loss = cross_entropy(noisy_labels(tf.ones_like(real_output), 0.05), real_output) # flip labels with 5% probability
    
        # loss when predict a fake image is fake
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        # fake_loss = cross_entropy(smooth_negative_labels(tf.zeros_like(fake_output)), fake_output)
    
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(self, fake_output):
        # generator's success is discriminator's loss when predict a fake image as real
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
        
    
    # #### Define Training loop
    # use of tf.function causes the function to be "compiled".
    @tf.function
    def train_step(self, images, batch_size):    
        seed = tf.random.normal([batch_size, self.seed_dim])
    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:    
            generated_images = self.GENERATOR(seed, training=True)
    
            # update the discriminator with separate batches of real and fake images rather than combining into a single batch
            real_output = self.DISCRIMINATOR(images, training=True)
            fake_output = self.DISCRIMINATOR(generated_images, training=True)
    
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
            gradients_of_generator = gen_tape.gradient(gen_loss, self.GENERATOR.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.DISCRIMINATOR.trainable_variables)
    
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.GENERATOR.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.DISCRIMINATOR.trainable_variables))
        
        return gen_loss,disc_loss
    
    
    def train(self, dataset, startepoch, epochs, batch_size):
      start = time.time()
       
      if startepoch == 0:
        # df = pd.DataFrame(columns=['epoch', 'batch', 'g_loss', 'd_loss'])
        df = pd.DataFrame.from_dict({'epoch': [], 'batch': [], 'g_loss': [], 'd_loss': []}) # for early version of Pandas
      else:
        df = pd.read_pickle(self.output_path + 'train_loss.pkl')
    
      for epoch in range(startepoch, startepoch + epochs):
        epoch_start = time.time()
    
        gen_loss_list = []
        disc_loss_list = []
        batch = 0
        for image_batch in dataset:
          batchloss = self.train_step(image_batch, batch_size)      
          gen_loss_list.append(batchloss[0])
          disc_loss_list.append(batchloss[1])
          df.loc[len(df.index)] = [epoch,batch,batchloss[0].numpy(),batchloss[1].numpy()]
          batch += 1
    
        g_loss = sum(gen_loss_list) / len(gen_loss_list)
        d_loss = sum(disc_loss_list) / len(disc_loss_list)
    
        # Save the model every 15 epochs
        if (epoch != 0) and (epoch % 15 == 0):
          # CHECKPOINT.save(file_prefix = os.path.join(output_path, "checkpt"))
          self.save_models(epoch)
          df.to_pickle(self.output_path + "train_loss.pkl")
    
        # Produce and save images during training
        output = self.output_path + 'gen_img_epoch{:04d}.png'.format(epoch)
        self.generate_and_save_images(output)
    
        epoch_elapsed = time.time()-epoch_start
        print (f'Epoch {epoch}, generator_loss={g_loss},discriminator_loss={d_loss}, run_time={self.hms_string(epoch_elapsed)}')
    
      elapsed = time.time()-start
      print (f'Total training time: {self.hms_string(elapsed)}')

    def hms_string(self, sec_elapsed):
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60.
        return "{}:{:>02}:{:>05.2f}".format(h, m, s)
    
    def save_models(self, epoch):
        self.GENERATOR.save(self.output_path + 'generator'+ str(epoch))
        self.DISCRIMINATOR.save(self.output_path + 'discriminator'+str(epoch))
        # save another copy of the latest one for convinence working with google drive
        self.GENERATOR.save(self.output_path + 'generator')
        self.DISCRIMINATOR.save(self.output_path + 'discriminator')
    
    def load_pretrained_models(self, epoch=0):
      path = self.output_path
      if epoch == 0:
        gname = "generator"
        dname = "discriminator"
      else:
        gname = "generator" + str(epoch)
        dname = "discriminator"+ str(epoch)
    
      # load_weights?
      GENERATOR = load_model(os.path.join(path, gname), compile=False)
      DISCRIMINATOR = load_model(os.path.join(path, dname), compile=False)
    
      GENERATOR.compile(optimizer='adam', loss='binary_crossentropy')
      DISCRIMINATOR.compile(optimizer='adam', loss='binary_crossentropy')
    
      return GENERATOR, DISCRIMINATOR

    def get_models(self, startepoch):
        if startepoch == 0:
           # build models
            model_of_choice = self.image_shape[0]
            if model_of_choice == 32:        
                GENERATOR = self.build_generator_32()
                DISCRIMINATOR = self.build_discriminator_32(self.image_shape)
            else: 
                GENERATOR = self.build_generator()
                DISCRIMINATOR = self.build_discriminator(self.image_shape)
    
            GENERATOR.compile(optimizer='adam', loss='binary_crossentropy')
            DISCRIMINATOR.compile(optimizer='adam', loss='binary_crossentropy')
        else:
           # reload trained models.
           GENERATOR, DISCRIMINATOR = self.load_pretrained_models()
    
        return GENERATOR, DISCRIMINATOR
    
    def generate_and_save_images(self, output):
        # set training to False so all layers run in inference mode (batchnorm)
        generated_images = self.GENERATOR(self.PREVIEW_SEED, training=False)  
        # scale from [-1,1] to [0,1]
        generated_images = (generated_images + 1) / 2.0
    
        fig = plt.figure(figsize=(4, 4))
        for i in range(generated_images.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated_images[i])
            # plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')      
            # plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5)
            plt.axis('off')
    
        plt.savefig(output)
        plt.show()

    def view_generated_sample(self):
        # an initial sample view
        noise = tf.random.normal([1, self.seed_dim])
        generated_image = self.GENERATOR(noise, training=False)
        # scale from [-1,1] to [0,1]  
        generated_image = (generated_image.numpy() + 1) / 2.0
        generated_image = generated_image.astype('float32')
    
        self.view_img_grid(generated_image, 'A Sample from Generator', 1)
        decision = self.DISCRIMINATOR(generated_image)
        return decision
    
    def view_orig_images(self, dataset, grid_size = 4):
        """
        plot images in a nxn grid
        dataset: tensorflow dataset in range [-1,1]
        grid_size: nxn grid containing images
        """     
        # get one batch of data
        dataset = next(iter(dataset))
        imgs = dataset.numpy()[:grid_size * grid_size]    
        # scale from [-1,1] back to [0,1] for dispaly
        imgs = (imgs + 1) / 2.0    
        
        self.view_img_grid(imgs, 'Original Training Images', grid_size)
            
    def view_img_grid(self, imgs, title='', grid_size = 4):    
        fig = plt.figure(figsize = (6, 6))
        plt.title(title)
        
        for i in range(grid_size * grid_size):
            plt.axis("off")
            fig.add_subplot(grid_size, grid_size, 1 + i) #(ROW,COLUMN,POSITION)
            plt.tight_layout()
            plt.imshow(imgs[i])
        plt.show()
        