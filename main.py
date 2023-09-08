# Importing Custom built fucntions
from yolo_func_utils import read_anchors, read_classes, preprocess_image, generate_colors
from yolo_head import yolo_head
from yolo_eval import yolo_eval
from yolo_draw_boxes import draw_boxes

# # Deep learning libraries (Tensorflow Framework)
# from tensorflow.keras.models import load_model

# Image Processing Libraries
import cv2

# Library for downloading model from google drive
import gdown

# Library for mathematical operations on multidimensional arrays
import numpy as np

# Libraries required for deployment (Done using Streamlit)
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoHTMLAttributes
import av


# # Initiallizing Variables
# # google drive link to the model
# url = 'https://drive.google.com/uc?id=1D3peEMoCqG1IdD6hYrHdSSBM6Bsdw2eP'
# output = 'yolo.h5'  # output location directory
# # reading class names stored in prediction_classes.txt
# class_names = read_classes("prediction_classes.txt")
# # reading anchors and their dimensions stored in prediction_anchors.txt
# anchors = read_anchors("prediction_anchors.txt")
# model_image_size = (608., 608.)  # Same as yolo_model input layer size
# # Selecting random colors for bounding boxes of different classes
# colors = generate_colors(class_names)

#---------------------------------------------------------------------------------------------------------------------------------------------------------
# st.write('Caution : For first-time usage, App can take some time to load (ps : for Downloading the Model)')
# # Making the Function decorator to memoize(cache) function execution of loading the model
# @st.cache
# def load_model_cached(url, output):

#     # step 1 : downloading the model.h5 file from the google drive url and storing to the 'output' location
#     gdown.download(url, output, quiet=False)

#     # step 2 : returning the model loaded using the tf.keras.load_model function
#     return(load_model('yolo.h5', compile=False))


# # storing the yolo model in yolo_model
# yolo_model = load_model_cached(url, output)


# # streamlit-webrtc Callback class for all the required frame processing for object detection
# # Input -> frame : input frame captured by webrtc-streamer
# # Output -> frame : processed input frame for object detection
# class VideoProcessor:

#     # taking variable input from outside the callback
#     def __init__(self) -> None:
#         self.max_boxes = 5

#     def recv(self, frame):

#         # step 1 : converting frame ('VideoFrame' Object  of 'pyAV' package) to numpy array
#         frm = frame.to_ndarray(format='bgr24')
#         # step 2 : resizing the ndarray to (1200,900)
#         image = cv2.resize(frm, (1200, 900), interpolation=cv2.INTER_LINEAR)
#         # step 3 : preprocessing the image for model implementation
#         image, image_data = preprocess_image(image, model_image_size=(608, 608))
#         # step 4 : Applying the model to image_data
#         yolo_model_outputs = yolo_model(image_data)
#         # step 5 : converting the model encoded output to more workable form
#         yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
#         # step 6 : finding the classes, boxes and their scores for the best detections
#         out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [900.0, 1200.0], self.max_boxes, 0.5, 0.5)
#         # step 7 : drawing the bounding boxes of predicted classes and objects
#         draw_boxes(image, out_scores, out_boxes,
#                    out_classes, class_names, colors)
#         # step 8 : returning the processed frame as a 'VideoFrame' Object
#         return av.VideoFrame.from_ndarray(image, format='bgr24')



#---------------------------------------------------------------------------------------------------------------------------------------------------------

# Streamlit Web-Based Real-Time Video Processing App Hosted on Streamlit Cloud

# Packages Used:
# streamlit : Main Package
# streamlit-webrtc : A custom component of Streamlit which deals with real-time video and audio streams
# opencv : Image processing


#settimng up background image for app
st.markdown(
   f'''
   <style>
   .stApp {{
             background: url("https://img.freepik.com/premium-photo/abstract-communication-technology-network-concept_34629-641.jpg?w=1380");
             background-size: cover
         }}
   </style>
   ''',
   unsafe_allow_html=True)


#creating containers for different sections of app
header = st.container()
app = st.container()
video = st.container()
model_intro = st.container()
model_details = st.container()
yolo_summary = st.container()
references = st.container()
side_bar = st.sidebar

with header:
    st.title('Real Time Object Detection')
    st.markdown("""---""")

    st.write('Object detection is an advanced form of image classification where a neural network predicts objects in an image and points them out in the form of bounding boxes.\nObject detection thus refers to the detection and localization of objects in an image that belong to a predefined set of classes.')
    st.markdown("""---""")

with side_bar:
    st.subheader('Predictable classes')
    st.markdown("""---""")
    for name in class_names:
        st.write(name) 

# with app:

#     st.subheader('The App')

#     # streamlit-webrtc requires callbacks to process image and audio frames which is one major
#     # difference between OpenCV GUI and streamlit-webrtc
#     ctx = webrtc_streamer(key='key',
#                         # class object passed to video_processor_factory for video processing
#                         video_processor_factory=VideoProcessor,
#                         # rtc_configuration parameter to deploy the app to the cloud
#                         rtc_configuration=RTCConfiguration(
#                             {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
#                         # muting the audio input
#                         video_html_attrs=VideoHTMLAttributes(
#                             autoPlay=True, controls=True, style={"width": "100%"}, muted=True),
#                         )

#     if ctx.video_processor:
#         ctx.video_processor.max_boxes = st.slider(
#             'Max boxes to predict', min_value=1, max_value=10, value=5, step=1)
#         # ctx.video_processor.score_threshold = st.slider(
#         #     'Score Threshold ', min_value=0.0, max_value=1.0, value=.5, step=.1)
#     st.markdown("""---""")

with video:

    st.subheader('Sample Video')
    video = open('sample.mp4','rb')
    st.video(video)


with model_intro:
    st.subheader('About YOLO ')
    st.write('YOLO (“You Only Look Once”) is an effective real-time object recognition algorithm, first described in the seminal 2015 paper by Joseph Redmon et al.Compared to the approach taken by object detection algorithms before YOLO, which repurpose classifiers to perform detection, YOLO proposes the use of an end-to-end neural network that makes predictions of bounding boxes and class probabilities all at once.')

    st.markdown('**YOLO Architecture**')
    st.write('Inspired by the GoogleNet architecture, YOLO’s architecture has a total of 24 convolutional layers with 2 fully connected layers at the end. ')
    st.image('images/yolo_architecture.png')
    st.markdown("""---""")

    st.subheader('Yolo Working')
    st.write('The YOLO algorithm works by dividing the image into N grids, each having an equal dimensional region of SxS. Each of these N grids is responsible for the detection and localization of the object it contains.Correspondingly, these grids predict B bounding box coordinates relative to their cell coordinates, along with the object label and probability of the object being present in the cell.This process greatly lowers the computation as both detection and recognition are handled by cells from the image, but—It brings forth a lot of duplicate predictions due to multiple cells predicting the same object with different bounding box predictions.YOLO makes use of Non Maximal Suppression to deal with this issue.')
    st.markdown("""---""")


with model_details:

    st.subheader('Model Details ')

    st.markdown('**Inputs and Outputs**')
    st.text('''
    ~ The input is a batch of images, and each image has the shape (m,608,608,
    ~ The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers  (p_c,b_x,b_y,b_h,b_w,c) as explained above. If you expand  into an 80-dimensional vector, each bounding box is then represented by 85 numbers. 
    ''')

    st.markdown('**Anchor Boxes**')
    st.text('''
    ~ Anchor boxes are chosen by exploring the training data to choose reasonable height/weight ratios that represent the different classes.
    ~ This model used 5 Anchor boxes (to cover 80 classes)
    ~ The dimension for anchor boxes is the second to last dimension in the ecoding :(m,n_h,n_w,anchors,classes)
    ~ The YOLO architecture is :
    Image (m,698,608,3) -> Deep CNN -> Encoding (m,19,19,5,85)
    ''')

    
    st.markdown('**Encoding**')
    st.text('''If the midpoint of an object falls into a grid cell , that grid cell is responsible for detecting that object.Since we have used 5 anchor boxes , each of the 19x19 grid cell gives thus encodes information about 5 boxes, Anchor boxes defined only by their weight and height.For simplicity , we'll flatten the last two dimensions fo the shape (19,19,5,85) encoding so the output of the Deep CNN is (19,19,425)
    ''')
    st.image('images/encoding.png')


    st.markdown('**Class Score**')
    st.text('''
    ~ For each box of each cell we compute the following element wise product and extract a probability that the box contains a certain class.
    ~ The class score is score(c,i) = p_c x c_i: the probability that their is an object p_c times the probability that the object is a certain class c_i.
    ~ Then we get rid of boxes with a low class score.The box is not very confident about detecting a class , either due to low probability of any object or low probability of this particular class.
    ''')
    st.image('images/class_score.PNG')

    st.markdown('**Non-Max Suppresion**')
    st.text('''
    ~ Even after thresholding on the basis of class scores we are still left with too many boxes, so we use non-max suppresion.
    ~ In this selection model we select only one box when several boxes overlap with each other and detect the same object.
    ~NMS uses the very important function called 'Intersection over Union' or IOU.
    ''')

    st.image('images/nms.png')
    st.markdown("""---""")


with yolo_summary:
    st.subheader('Yolo Summary')
    st.text('''
    ~ YOLO is a state-of-the-art object detection model that is fast and accurate
    ~ It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume.
    ~ The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
    ~ You filter through all the boxes using non-max suppression. Specifically:
    ~ Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
    ~ Intersection over Union (IoU) thresholding to eliminate overlapping boxes
    ~ Training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation.
    ~ Here we used previously trained model parameters.
    ''')
    st.markdown("""---""")


with references:
    st.subheader('Refernces')
    checkbox = st.checkbox("Show references")
    if(checkbox):
        st.write("""
        1. V7 labs (https://www.v7labs.com/blog/yolo-object-detection) for studying mathematical theory behind YOLO model.
        2. Official Yolo Website (https://pjreddie.com/darknet/yolo/) for downloading pretrained weights and configuration file of yolo version 2.
        3. Coursera DeepLearning Specialization (https://www.coursera.org/specializations/deep-learning) for project implementation.
        4. Dr. Andrew NG (My role Model for ML/DL)
        """)