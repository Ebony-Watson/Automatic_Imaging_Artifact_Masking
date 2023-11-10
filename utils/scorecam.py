import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Activation, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

def softmax(x):
    """
    Function taken from the tabayashi0117 implementation of the Score-CAM framework, available at: https://github.com/tabayashi0117/Score-CAM and in software folder.
    """
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

def ScoreCam(model, img_array, layer_name, max_N=-1):
    """
    Build Score-CAM framework for producing saliency maps highlighting image regions most influential for prediction of the target class from the wrapped model.
    
    Function taken from the tabayashi0117 implementation of the Score-CAM framework, available at: https://github.com/tabayashi0117/Score-CAM and in software folder.
    """
    cls = np.argmax(model.predict(img_array))
    act_map_array = Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(img_array)
    
    # extract effective maps
    if max_N != -1:
        act_map_std_list = [np.std(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])]
        unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
        max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
        act_map_array = act_map_array[:,:,:,max_N_indices]

    input_shape = model.layers[0].output_shape[0][1:]  # get input shape
    # 1. upsample to original input size
    act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape[:2], interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
    # 2. normalize the raw activation value in each activation map into [0, 1]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)
    # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(img_array)
        for k in range(3):
            masked_input[0,:,:,k] *= act_map_normalized
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)
    # 4. feed masked inputs into CNN model and softmax
    pred_from_masked_input_array = softmax(model.predict(masked_input_array, batch_size=12))
    # 5. define weight as the score of target class
    weights = pred_from_masked_input_array[:,cls]
    # 6. get final class discriminative localization map as linear weighted combination of all activation maps
    cam = np.dot(act_map_array[0,:,:,:], weights)
    cam = np.maximum(0, cam)  # Passing through ReLU
    cam /= np.max(cam)  # scale 0 to 1.0
    
    return cam

def read_and_preprocess_img(path, size=(576,576)):
            img = load_img(path, color_mode='grayscale', target_size=size)
            img = img_to_array(img)
            img = np.uint8(img/256)
            img = np.repeat(img[:,:, :], 3, -1)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            return img
        
def edge_map(img):
    """
    Produce an edge map of original image for use in fine-tuning Score-CAM saliency map.
    """
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) # calculate horizontal image gradients
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) # calculate vertical image gradients
    grad = np.sqrt(dx ** 2 + dy ** 2) #Get gradient magnitude
    grad = cv2.dilate(grad,kernel=np.ones((5,5)), iterations=1) 
    grad -= np.min(grad) # scale 0. to 1.
    grad /= np.max(grad)  
    return grad