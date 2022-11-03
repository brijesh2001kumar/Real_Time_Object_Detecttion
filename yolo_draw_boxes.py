# Importing required Libraries 
import numpy as np
import cv2

"""

Draw the bounding boxes with the label around the detecteted objects 

Arguments:
image -- opencv image object 'bgr24' 
out_boxes -- list of predicted boxes after applying all the filters and thresholds
out_scores -- list of predicted scores of the final predicted boxes
out_classes -- list of the output classes arranged in desceding order of prediction probability
class_names -- list of the class names stored as string
colors --  list of the different colors assigned to different classes

Returns:
Updated image with bounding box and label

"""


def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):

    # defining font and image shape variables
    (height,width,channels) = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    # looping through the predicted classes one at a time 
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
        
        # desinging the label of bnd box
        label = '{} {:.2f}'.format(predicted_class.upper(), score)
        # getting the labels size (height,width)
        label_size, baseline = cv2.getTextSize(label, font, 1, 2)  

        # computing the top, left, bottom, right corner coordinates of the bounding box
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
        right = min(width, np.floor(right + 0.5).astype('int32'))
        
        #computing the positioning of the label on bnd box
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # Drawing the bounding box and label on the image
        cv2.rectangle(image,(left, top),(right, bottom),color = colors[c],thickness = 3)
        cv2.rectangle(image,tuple(text_origin-[5,5]),tuple(text_origin + label_size),colors[c],-1)
        cv2.putText(image, label, tuple([left, top]), font, 1, (0, 0, 0),thickness = 2)
        cv2.putText(image, str(i) + ' : ' + label, tuple([10,label_size[1] + i*label_size[1] + i*5 + 10]), font, 1, (0, 0, 0),2)
