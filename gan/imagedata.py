# ### Imports and setup

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

import numpy as np
import zipfile
import os 
from tqdm import tqdm

class imagedata:
    
    def __init__(self, collection, datapath):
        self.buffer_size = 60000        
        #self.data_path = 'data/' + collection        
        self.data_path = datapath + collection
#         return self.load_dataset(datatype, batch_size, image_shape)
    
    # load dataset as TensorFlow Dataset object, 
    # so the data can be quickly shuffled and divided into the appropriate batch sizes for training.
    def load_dataset(self, datatype, batch_size, image_shape):
        buffer_size = self.buffer_size, 
        
        if datatype == 'npz':
            train_dataset = self.load_processed_data(batch_size)
        elif datatype == 'zip':            
            zipfile = self.data_path + '.zip'
            with zipfile.ZipFile(zipfile,"r") as zip_ref:                
                zip_ref.extractall(self.data_path)
    
            # train_dataset = load_image_data(filedir, batch_size, image_shape)
            np_data = self.save_dataset(self.data_path)
            train_dataset = tf.data.Dataset.from_tensor_slices(np_data).shuffle(buffer_size).batch(batch_size,drop_remainder=True)
        elif datatype == 'img':            
            np_data = self.save_dataset(self.data_path)
            train_dataset = tf.data.Dataset.from_tensor_slices(np_data).shuffle(buffer_size).batch(batch_size,drop_remainder=True)
            
        return train_dataset
    
    def save_dataset(self, sourcedir):
        result_imgshape = self.image_shape
        training_data = []  
        imgfiles   = np.sort(os.listdir(sourcedir))
        for filename in tqdm(imgfiles):      
            path = os.path.join(sourcedir,filename)
            try: 
                image = Image.open(path).resize(result_imgshape[:2], Image.ANTIALIAS).convert('RGB')
                training_data.append(np.asarray(image))
            except: #skip bad images
                pass      
      
        training_data = np.reshape(training_data,(-1, result_imgshape[0], result_imgshape[1], result_imgshape[2]))
        training_data = training_data.astype(np.float32)
        # Normalize the images to [-1, 1]. RGB pixel scale for original images is from 1–255
        training_data = training_data / 127.5 - 1.
    
        # Saving training image binary
        outputfile = self.data_path + ".npz"  # size much smaller than .npy
        np.savez_compressed(outputfile, training_data)
      
        return training_data
    
    # load original images to Tensorflow dataset
    def load_image_data(self, filedir, batch_size, image_shape):         
        datasets = image_dataset_from_directory(
                            filedir,
                            validation_split = None,
                            image_size = image_shape[:2],
                            shuffle=True,
                            batch_size = batch_size)
    
        # scale pixel values to [-1,1]
        normalization_layer = Rescaling(1./127.5, offset=-1)
        datasets = datasets.map(lambda x, y: (normalization_layer(x), y))
        
        # need image_batch only. this is only one batch, need all the batchs
        image_batch, label_batch = next(iter(datasets))
    
        return image_batch
    
    # load pre-processed dataset
    def load_processed_data(self, batch_size):
        buffer_size = self.buffer_size
        
        np_data = np.load(self.data_path + '.npz')['arr_0']
     
        # use TensorFlow Dataset object to hold the images
        train_dataset = tf.data.Dataset.from_tensor_slices(np_data).shuffle(buffer_size).batch(batch_size,drop_remainder=True)
    
        return train_dataset
    