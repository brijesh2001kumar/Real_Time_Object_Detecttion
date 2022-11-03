# Importing required Libraries 
import numpy as np
from PIL import ImageFont,ImageDraw

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
    
    # import the font from directory "FiraMono-Regular.otf"
    font = ImageFont.truetype(font='FiraMono-Regular.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

    thickness = (image.size[0] + image.size[1]) // 300  # thickness of the edge of bnd box

    # looping through the predicted classes one at a time 
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        # desinging the label of bnd box
        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)  #intantiating ImageDraw.Draw component for drawing on image
        label_size = draw.textsize(label, font)  # getting the labels size (height,width)

        # computing the top, left, bottom, right corner coordinates of the bounding box
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        
        #computing the positioning of the label on bnd box
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # Drawing the bounding box and label on the image
        for j in range(thickness):
            draw.rectangle([left + j, top + j, right - j, bottom - j], outline=colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            draw.text(np.array([image.size[0]-label_size[0], 0 + i*label_size[1]]),
                      str(i) + ' : ' + label,
                      fill=(0, 0, 0), 
                      font=font)
        del draw

