import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input,ResNet50
from tensorflow.keras.layers import Dense, Input, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def data_resize(width=576,height=576,ftype='.tiff',load_dir=None,save_dir=None):
    for file in os.listdir(load_dir):
        image_path = os.path.join(load_dir,file)
        image_n = file.split('.')[0]
        im = cv2.imread(image_path, -1)
        im = cv2.resize(im, (width, height),interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(save_dir, f'{image_n}_{width}x{height}{ftype}'), im)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,labels, batch_size=32, dim=(576,576), n_channels=3,
                 n_classes=2, shuffle=True, PATH=None, Augment=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.PATH = PATH
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = Augment
        self.on_epoch_end()
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_labels_temp = [self.labels[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_labels_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_augmentation(self, img):
        
        action_list = ['random_shift', 'random_h_flip', 'random_v_flip',  'rotate_90',
                       'none','random_brightness','random_contrast'] #

        # Random values to select an operation
        operations = np.random.random(7).tolist()
        maximum = operations.index(max(operations))
        op = action_list[maximum]
        if op == 'random_shift':
            img = tf.keras.preprocessing.image.random_shift(img, 0.2, 0.3)
        elif op == 'random_h_flip':
            img = tf.image.random_flip_left_right(img)
        elif op == 'random_v_flip':
            img = tf.image.random_flip_up_down(img)
        elif op == 'random_brightness':   
            img = tf.image.random_brightness(img, 0.2)
        elif op == 'rotate':       
            img = tf.image.rot90(img)
        elif op == 'random_contrast':     
            img = tf.image.random_contrast(img, 0.2, 0.5)
        else:
            return img
        
        return img
    
    def __data_generation(self, list_IDs_temp, list_labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,),dtype=int)
        PATH = self.PATH
        # Generate data
            
        for i, (ID, lab) in enumerate(zip(list_IDs_temp,list_labels_temp)):
            # Store sample
                img = load_img('/'.join([PATH, ID]), color_mode='grayscale', target_size=self.dim)
                img = img_to_array(img)
                img = np.uint8(img/256)
                img = np.repeat(img[:,:, :], 3, -1)
                #img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                if self.augment is True:
                    img = self.__data_augmentation(img)
                X[i,] = img
            
                y[i] = lab
                
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)



class DataGenerator_Pred(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=12, dim=(576,576), n_channels=3, shuffle=True, PATH=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.PATH = PATH
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
                
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        PATH = self.PATH
        # Generate data
            
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
                img = load_img('/'.join([PATH, ID]), color_mode='grayscale', target_size=self.dim)
                img = img_to_array(img)
                img = np.uint8(img/256)
                img = np.repeat(img[:,:, :], 3, -1)
                #img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                X[i,] = img
                 
        return X
    
    
def build_ResNet(img_size=(576,576),weights=None):
    model = ResNet50(include_top=True, weights=weights,input_tensor=Input(shape=(img_size[0],img_size[1],3)))

    x = model.layers[-98].output
    x = Activation('relu', name="act_last")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, name="dense_out")(x)
    outputs = Activation('softmax')(x)

    model = Model(model.input, outputs)
    # model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.002,amsgrad=True),
                  metrics=['accuracy'])
    return model

