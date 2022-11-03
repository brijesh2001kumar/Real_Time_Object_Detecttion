#Importing required Libraries
import tensorflow as tf
import numpy as np


"""
reshaping the yolo model encodings to usable form
        (None,19,19,425) -> (None,19,19,5,80)

Inputs:
feats -- tensor([None,19,19,425]) yolo model output encodings
anchors -- list of shaep (5,2) containing dimensions of 5 bounding boxes
num_classes -- (int) number of classes = 80

Outputs:
box_xy -- tensor([None,19,19,2]) midpoint coordinates of bnd boxes (0 >= x,y <= 1)
box_wh -- tensor([None,19,19,2]) widht and height of the bounding boxes (w,h >= 0)
box_confidence -- tensor([None,19,19,1]) probability of detecting an object in the given grid cell (0 >= p <= 1)
box_class_probs -- tensor([None,19,19,80]) probabilities of the respective 80 classes (0 >= c <= 1)
"""

def yolo_head(feats, anchors, num_classes):
    
    anchors_tensor = tf.reshape(tf.Variable(anchors, dtype='float32'), [1, 1, 1, 5, 2])  

    yolo_model_outputs = feats

    array = yolo_model_outputs.numpy()             #converting tensor to numpy object for easy computation
    new_array = np.resize(array, (1,19,19,5,85))   #resizing from (None,19,19,425) -> (None,19,19,5,80)
    yolo_outputs = tf.convert_to_tensor(new_array) #converting back to tensor from numpy object 

    box_xy = tf.sigmoid(yolo_outputs[..., :2])              # Applying sigmoid to first 2 columns for getting x an y s.t. 
    box_wh = tf.exp(yolo_outputs[..., 2:4])                 # Applying exponential to first 3 and 4  columns for getting w an h s.t. (w,h >= 0) 
    box_confidence = tf.sigmoid(yolo_outputs[..., 4:5])     # Applying sigmoid to first 5 columns for getting x an y s.t. (0 >= p <= 1)
    box_class_probs = tf.nn.softmax(yolo_outputs[..., 5:])  # Applying softmax to first 6 to 85 columns for getting x an y s.t. (0 >= c <= 1)


    
    #creating conv_index tensor for scaling the x and y ( initialy computed w.r.t. grid cells ) 
    #coodinates with respect to the whole image (19 x 19 grid cells)
    #
    #instance of conv_index :-
    #conv_index = [[[[0,0],[0,1],[0,2],.........,[0,18]],
    #            [[1,0],[1,1],[1,2],...........,[1,18]],
    #            .......................................
    #            .......................................
    #            [[18,0],[18,1],[18,2],........,[18,18]]]]

    array = np.arange(0,19,1,dtype='float32')
    array = np.tile(array,(19,1))
    array_t = np.transpose(array)
    array2 = np.stack((array,array_t), -1)
    array3 = np.expand_dims(array2, axis=2)
    conv_index = np.expand_dims(array3, axis=0)
    conv_index = tf.convert_to_tensor(conv_index)


    #creating conv_dims tensor for scaling down the x,y,w,h w.r.t. the whole image 
    #
    #conv_dims = tensor(shape=[1,1,1,1,2], [19.0,19.0])

    dims = np.array([19.0,19.0],dtype='float32') 
    dims = np.reshape(dims,(1,1,1,1,2))
    conv_dims = tf.convert_to_tensor(dims)

    # applying operations to get final box_xy and box_wh 
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs