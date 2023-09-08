# Importing the required Libraries and packages
import numpy as np
import tensorflow as tf
import colorsys
import random
import cv2

"""
Convert YOLO box prediction to bounding box corners.
"""

def yolo_boxes_to_corners(box_xy, box_wh):

    # computing the top left corner co-ordinates of bounding boxes
    box_mins = box_xy - (box_wh / 2.0)
    box_maxes = box_xy + (box_wh / 2.0)
    
    return tf.keras.backend.concatenate([
        box_mins[...,1:2],  #y_min
        box_mins[...,0:1],  #x_min
        box_maxes[...,1:2],  #y_max
        box_maxes[...,0:1]   #x_max
    ])



""" 
Scales the predicted boxes in order to be drawable on the image
"""

def scale_boxes(boxes, image_shape):
    
    # boxes : shape [1,4] with each entry between [0,1]
    # height & width : scaling factor for boxes 
    height = image_shape[0]
    width = image_shape[1]
    image_dims = tf.stack([height, width, height, width])
    image_dims = tf.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims 
    
    return boxes


"""
read classes names as string from prediction_class.text
"""

def read_classes(classes_path):
    with open(classes_path, "r", encoding='utf-8', errors="ignore") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

"""
read anchor dimensions as float dtype from prediction_anchors.text
"""

def read_anchors(anchors_path):

    with open(anchors_path,"r", encoding='utf-8', errors="ignore") as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

"""
preprocessing the input image 
"""

def preprocess_image(image, model_image_size):

    resized_image = cv2.resize(image, tuple(reversed(model_image_size)), interpolation=cv2.INTER_LINEAR)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    return image, image_data

"""
generate different colors for each class 
"""

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors
