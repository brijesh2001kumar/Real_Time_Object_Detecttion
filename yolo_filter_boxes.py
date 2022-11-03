import tensorflow as tf


"""
Filter boxes by thresholding on object and class confidence

Arguments:
boxes -- tensor of shape (19,19,5,4) containing the midpoint and dimensions (b_x, b_y, b_h, b_w) for each of the 5 boxes.
box_confidence -- tensor of shape (19,19,5,1) containing Pc for each of the 5 boxes predicted in each of the 19x19 grid.
box_class_probs -- tensor of shape (19,19,5,80) containing "class probabilites" for each of the 80 classes for each of the 5 boxes per cell.
threshold -- real value , tf [highest class probability score < threshold], then get rid of the corresponding box

Returns:
scores -- tensor of shape (None,), containig the class probability score for selected boxes
boxes -- tensor of shape (None, 4), containing (b_x,b_y,b_h,b_w) coodinates of selected boxes
classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

"""

def yolo_filter_boxes (boxes, box_confidence, box_class_probs, threshold = 0.6):
    
    #step 1 : compute box scores     
    box_scores = box_confidence * box_class_probs
    
    #step 2 : find the box_classes using the max box_scores, keeping track of the corresponding score
    box_classes = tf.math.argmax(box_scores, axis = -1)
    box_class_scores = tf.math.reduce_max(box_scores, axis = -1, keepdims = False)
    
    #step 3 : create a filetring mask based on 'box_class_scores' by using 'threshold' 
    filtering_mask = (box_class_scores >= threshold)
    
    #step 4 : Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes