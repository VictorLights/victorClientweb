import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import time ,sys
from streamlitp_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
import cv2
import time
import sys
import yaml
from yaml.loader import SafeLoader


class YOLO_Pred():
    def __init__(self,onnx_model,data_yaml):
        # load YAML
        with open(data_yaml,mode='r') as f:
            data_yaml = yaml.load(f,Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        
        # load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    
        
    def predictions(self,image):
        
        row, col, d = image.shape
        # get the YOLO prediction from the the image
        # 1 convert image into square image (array)
        max_rc = max(row,col)
        input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
        input_image[0:row,0:col] = image
        # 2 get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WH_YOLO,INPUT_WH_YOLO),swapRB=True,crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward() # detect prediction from YOLO

        # Non Maximum Supression
        # filter detection based on confidence (0.4) and probability score (0.25)
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # width and height of the image (input_image)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/INPUT_WH_YOLO
        y_factor = image_h/INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4] 
            if confidence > 0.4:
                class_score = row[5:].max() # maximum probability of the 20 objects
                class_id = row[5:].argmax() # get the index position at which max probabilty occur

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # construct bounding from four values
                    # left, top, width and height
                    left = int((cx - 0.5*w)*x_factor)
                    top = int((cy - 0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)

                    box = np.array([left,top,width,height])

                    # append values into the list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # clean
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # NMS
        #index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()
        index = np.array(cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)).flatten()
        


        # Draw the Bounding
        for ind in index:
            # extract bounding box
            x,y,w,h = boxes_np[ind]
            bb_conf = int(confidences_np[ind]*100)
            classes_id = classes[ind]
            print(len(self.labels))
    
            class_name = self.labels[classes_id]
            colors = self.generate_colors(classes_id)

            text = f'{class_name}: {bb_conf}%'

            cv2.rectangle(image,(x,y),(x+w,y+h),colors,2)
            cv2.rectangle(image,(x,y-30),(x+w,y),colors,-1)

            cv2.putText(image,text,(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(0,0,0),1)
            
        return image
    
    
    def generate_colors(self,ID):
        np.random.seed(10)
        colors = np.random.randint(150,255,size=(self.nc,3)).tolist()
        return tuple(colors[ID])


st.set_page_config(page_title="Computer Vision",
                   layout='wide',
                   page_icon='./images/FPTlogo.png')

st.title('Object Detection v1')
st.subheader("""
    Please Upload Image to get detections
    """)

with st.spinner('Please wait while your model is loading'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx',
                    data_yaml='./models/data.yaml')
    st.balloons()

def upload_image():
    # Upload Image
    image_file = st.file_uploader(label='Upload Image', type=['png','jpg','jpeg'])
    if image_file is not None:
        size_mb = image_file.size/(1024**2)
        file_details = {"filename":image_file.name,
                        "filetype":image_file.type,
                        "filesize": "{:,.2f} MB".format(size_mb)}
        #st.json(file_details)
        
        # validate file
        if file_details['filetype'] in ('image/png','image/jpeg'):
            #st.image(image_file, caption = "Uploaded Image")
            st.success('VALID IMAGE file type (png or jpeg)')
            return {"file":image_file,
                    "details":file_details}
        
        else:
            st.error('INVALID Image file type')
            st.error('Upload only png,jpg, jpeg')
            return None
   
        
def main():
    object = upload_image()
    
    if object:
        prediction = False
        image_obj = Image.open(object['file'])       
        
        col1 , col2 = st.columns(2)
        
        with col1:
            st.info('Preview of Image')
            st.image(image_obj)
            
        with col2:
            st.subheader('Check below for file details')
            st.json(object['details'])
            button = st.button('Get Detection from YOLO')
            if button:
                with st.spinner("""
                Geting Objets from image. please wait
                                """):
                    # convert obj to array
                    image_array = np.array(image_obj)
                    pred_img = yolo.predictions(image_array)
                    pred_img_obj = Image.fromarray(pred_img)
                    prediction = True
                
        if prediction:
            st.subheader("Predicted Image")
            st.caption("Object detection from YOLO V8 model")
            st.image(pred_img_obj)
    
    
    
if __name__ == "__main__":
    main()