import numpy as np
import cv2
import keras
from skimage.io import imread
import matplotlib.pyplot as plt
import tensorflow

from tensorflow.keras.optimizers import Adam #,schedules
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, Activation, Concatenate


def configuration(num_filters_start = 64,num_unet_blocks = 5,num_filters_end = 1,input_width = 576,input_height = 576,mask_width = 576,mask_height = 576,input_dim = 1,optimizer = Adam(lr=0.002),
                 loss = BinaryCrossentropy,initializer = HeNormal(),batch_size = 4, num_epochs = 150,metrics = ['BinaryIoU','accuracy'],dataset_path = None):
    ''' Get configuration for UNET. '''

    return dict(
        num_filters_start = num_filters_start,
        num_unet_blocks = num_unet_blocks,
        num_filters_end = num_filters_end,
        input_width = input_width,
        input_height = input_height,
        mask_width = mask_width,
        mask_height = mask_height,
        input_dim =  input_dim,
        optimizer = optimizer,
        loss = loss,
        initializer = initializer,
        batch_size = batch_size,
        num_epochs = num_epochs,
        metrics = metrics,
        dataset_path = dataset_path,
    )


'''
    U-NET BUILDING BLOCKS
'''
def compute_number_of_filters(block_number):
    '''
        Compute the number of filters for a specific
        U-Net block given its position in the contracting path.
    '''
    return configuration().get("num_filters_start")# * (2 ** block_number)

def conv_block(x, filters, last_block):
    '''
        U-Net convolutional block.
        Used for downsampling in the contracting path.
    '''
    config = configuration()

    # First Conv segment
    x = Conv2D(filters, (3, 3),\
            kernel_initializer=config.get("initializer"),padding='same')(x)
    x = Activation("relu")(x)

    # Second Conv segment
    x = Conv2D(filters, (3, 3),\
        kernel_initializer=config.get("initializer"),padding='same')(x)
    x = Activation("relu")(x)

    # Keep Conv output for skip input
    skip_input = x

    # Apply pooling if not last block
    if not last_block:
        x = MaxPool2D((2, 2), strides=(2,2))(x) # strides=(2,2)

    return x, skip_input

def contracting_path(x):
    '''
        U-Net contracting path.
        Initializes multiple convolutional blocks for 
        downsampling.
    '''
    config = configuration()

    # Compute the number of feature map filters per block
    num_filters = [compute_number_of_filters(index)\
            for index in range(config.get("num_unet_blocks"))]
    skip_inputs = []

    # Pass input x through all convolutional blocks and
    # add skip input Tensor to skip_inputs if not last block
    for index, block_num_filters in enumerate(num_filters):

        last_block = index == len(num_filters)-1
        x, skip_input = conv_block(x, block_num_filters,\
            last_block)

        if not last_block:
            skip_inputs.append(skip_input)

    return x, skip_inputs

def upconv_block(x, filters, skip_input, last_block = False):
    '''
        U-Net upsampling block.
        Used for upsampling in the expansive path.
    '''
    config = configuration()

    # Perform upsampling
    x = Conv2DTranspose(filters//2, (2, 2), strides=(2, 2), #
        kernel_initializer=config.get("initializer"), padding="same")(x)
    shp = x.shape

    # Crop the skip input, keep the center
   #cropped_skip_input = CenterCrop(height = x.shape[1],\
   #     width = x.shape[2])(skip_input)

    # Concatenate skip input with x
    concat_input = Concatenate(axis=-1)([skip_input, x])#cropped_skip_input

    # First Conv segment
    x = Conv2D(filters//2, (3, 3),#
        kernel_initializer=config.get("initializer"), padding="same")(concat_input)
    x = Activation("relu")(x)

    # Second Conv segment
    x = Conv2D(filters//2, (3, 3), #
        kernel_initializer=config.get("initializer"), padding="same")(x)
    x = Activation("relu")(x)

    # Prepare output if last block
    if last_block:
        x = Conv2D(config.get("num_filters_end"), (1, 1),activation='sigmoid',
            kernel_initializer=config.get("initializer"), padding="same")(x)

    return x

def expansive_path(x, skip_inputs):
    '''
        U-Net expansive path.
        Initializes multiple upsampling blocks for upsampling.
    '''
    num_filters = [compute_number_of_filters(index)\
            for index in range(configuration()\
                .get("num_unet_blocks")-1, 0, -1)]
    skip_max_index = len(skip_inputs) - 1

    for index, block_num_filters in enumerate(num_filters):
        skip_index = skip_max_index - index
        last_block = index == len(num_filters)-1
        x = upconv_block(x, block_num_filters,\
            skip_inputs[skip_index], last_block)

    return x

def training_callbacks():
    ''' Retrieve initialized callbacks for model.fit '''
    return [cp_cb,reduce_lr]#,es_cb

def build_unet():
    ''' Construct U-Net. '''
    config = configuration()
    input_shape = (config.get("input_height"),\
        config.get("input_width"), config.get("input_dim"))

    # Construct input layer
    input_data = Input(shape=input_shape)

    # Construct Contracting path
    contracted_data, skip_inputs = contracting_path(input_data)

    # Construct Expansive path
    expanded_data = expansive_path(contracted_data, skip_inputs)

    # Define model
    model = Model(input_data, expanded_data, name="U-Net")

    return model


def init_model(steps_per_epoch):
    '''
        Initialize a U-Net model.
    '''
    config = configuration()
    model = build_unet()

    # Retrieve compilation input
    loss_init = config.get("loss")()#(from_logits=True)
    metrics = config.get("metrics")
    num_epochs = config.get("num_epochs")

    # Construct LR schedule
    #boundaries = [int(num_epochs * percentage * steps_per_epoch)\
    #    for percentage in config.get("lr_schedule_percentages")]
    #lr_schedule = config.get("lr_schedule_class")(boundaries, config.get("lr_schedule_values"))

    optimizer_init = config.get("optimizer") #,learning_rate =0.002) # Init optimizer
    model.compile(loss=loss_init, optimizer=optimizer_init,metrics=metrics) # Compile the model
    #plot_model(model, to_file="unet.png") # Plot the model   
    #model.summary() # Print model summary

    return model



def generate_plot(img_input, mask_truth, mask_probs):
    ''' Generate a plot of input, truthy mask and probability mask. '''
    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(16, 6)

    axs[0].imshow(img_input, cmap='gray') # Plot the input image
    axs[0].set_title("Input image")
    
    axs[1].imshow(mask_truth) # Plot the truthy mask
    axs[1].set_title("True mask")
    
    predicted_mask = mask_probs#probs_to_mask(mask_probs) # Plot the predicted mask
    axs[2].imshow(predicted_mask)
    axs[2].set_title("Predicted mask")

    config = configuration() # Plot the overlay
    img_input_resized = tensorflow.image.resize(img_input, (config.get("mask_width"), config.get("mask_height")))
    axs[3].imshow(img_input_resized, cmap='gray')
    axs[3].imshow(predicted_mask, alpha=0.4)
    axs[3].set_title("Overlay")

    plt.show()
    
    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(576,576), n_channels=1,
                 n_classes=1, shuffle=True, PATH=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.PATH = PATH
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
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
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,*self.dim, self.n_channels),dtype=int)
        PATH = self.PATH
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = cv2.imread(PATH + '/images/' + ID, -1)#[:,:,:IMG_CHANNELS]
            img = (img/256).astype('uint8')
            img = np.expand_dims(img, axis=-1)
            X[i,] = img
            
            mask_ = imread(PATH + '/masks/' + ID.split('.tiff')[0] + '_Mask.tiff',-1) #'_Mask.tif'
            mask_ = cv2.resize(mask_ , (576,576), interpolation = cv2.INTER_AREA)
            threshold, mask_  = cv2.threshold(mask_ , 0.15, 1, cv2.THRESH_BINARY)
            mask_ = np.expand_dims(mask_, axis=-1) #resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True)
            # Store class
            y[i] = mask_

        return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)