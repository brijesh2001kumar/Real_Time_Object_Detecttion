import tensorflow as tf



"""
Apply non_max_suppression to set of boxes

Arguments:
scores : tensor of shape (None,) output of yolo_filter_boxes()
boxes : tensor of shape (None,4) output of yolo_filter_boxes()  that have been scaled to the image size
classes : tensor of shape (None,) output of yolo_filter_boxes()
max_boxes : integer, max. number of predicted boxes we'd like to have
iou_threshold : real value, 'Intersection over Union' threshold used for NMS filtering

Returns:
scores : tensor of shape (, None), predicted score for each box
boxes : tensor of shape (4, None), predicted box coordinates
classes : tensor of shape (, None), predicted class for each box
"""

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    
    #tensor to be used in tf.image.non_max_suppression()
    max_boxes_tensor = tf.Variable (max_boxes, dtype = 'int32')
    
    #Use tensorflow built-in funtion to get the list of indices corresponding to boxes to keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_output_size = max_boxes, iou_threshold = 0.5, name = None)
    
    #Use tensorflow built-in funtion to select only nms_indices from scores, boxes and classes
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices )
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes