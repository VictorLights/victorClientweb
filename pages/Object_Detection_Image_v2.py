import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
#import tensorflow as tf
import time ,sys
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
import cv2
import numpy as np
import time
import sys
import plotly.express as px
import random

   
st.set_page_config(page_title="OD Version 2.0",
                   layout='wide',
                   page_icon='./images/FPTlogo.png')
  
def object_detection_image():
    st.title('Object Detection Stats Visual v2')
    st.subheader("""
    Please Upload Image to get detections
    """)
    file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    if file!= None:
        img1 = Image.open(file)
        img2 = np.array(img1)

        st.image(img1, caption = "Uploaded Image")
        my_bar = st.progress(0)
        confThreshold =st.slider('Confidence', 0, 100, 50)
        nmsThreshold= st.slider('Threshold', 0, 100, 20)
        #classNames = []
        whT = 320
        
        url = "https://raw.githubusercontent.com/VictorLights/victorClientweb/main/coconames.txt"
        f = urllib.request.urlopen(url)
        classNames = [line.decode('utf-8').strip() for  line in f]
        #f = open(r'C:\Users\tangyifong\Downloads\victorlightsApp\victorClientweb\coconames.txt','r')
        #lines = f.readlines()
        #classNames = [line.strip() for line in lines]
        
        # Method with absolute path
        #config_path = r'/Users/tangyifong/Documents/victorlightsApp/victor_client_webapp/victorclient/config_n_weights/yolov3.cfg'
        #weights_path = r'/Users/tangyifong/Documents/victorlightsApp/victor_client_webapp/victorclient/config_n_weights/yolov3.weights'
        #net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        config_path = r'./config_n_weights/yolov3.cfg'  
        weights_path = r'./config_n_weights/yolov3.weights'
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Method with url
        # config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
        # weights_url = "https://pjreddie.com/media/files/yolov3.weights"
        # config_file = urllib.request.urlopen(config_url)
        # weights_file = urllib.request.urlopen(weights_url)
        # config_path = 'yolov3.cfg'
        # weights_path = 'yolov3.weights'
        # with open(config_path, 'wb') as f:
        #     f.write(config_file.read())
        # with open(weights_path, 'wb') as f:
        #     f.write(weights_file.read())
        
        
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        def findObjects(outputs,img):
            hT, wT, cT = img2.shape
            bbox = []
            classIds = []
            confs = []
            for output in outputs:
                for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > (confThreshold/100):
                        w,h = int(det[2]*wT) , int(det[3]*hT)
                        x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                        bbox.append([x,y,w,h])
                        classIds.append(classId)
                        confs.append(float(confidence))
        
            indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold/100, nmsThreshold/100)
            obj_list=[]
            confi_list =[]
                
                    
        
            #drawing rectangle around object
            for i in indices:
                i = i
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                # print(x,y,w,h)
                cv2.rectangle(img2, (x, y), (x+w,y+h), (240, 54 , 230), 2)
                #print(i,confs[i],classIds[i])
                obj_list.append(classNames[classIds[i]].upper())
                
                confi_list.append(int(confs[i]*100))
                cv2.putText(img2,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            df= pd.DataFrame(list(zip(obj_list,confi_list)),columns=['Object Name','Confidence'])
            object_counts = df['Object Name'].value_counts()
            if st.checkbox("Show Object's list" ):
                st.write(df)
                st.write('Number of objects detected: ')
                st.write(object_counts)
            if st.checkbox("Show Confidence bar chart" ):
                st.subheader('Bar chart for confidence levels')
                st.bar_chart(df["Confidence"])
                colorschart = ['#%06X' % random.randint(0, 0xFFFFFF) for i in range(len(object_counts))]
                fig = px.bar(object_counts, color=object_counts.index, color_discrete_sequence=colorschart)
                st.plotly_chart(fig, use_container_width=True)        
        
        
        st.subheader('Check below for file details')
        file_details = {"filename": file.name,
                "filetype": file.type,
                "filesize": "{:,.2f} MB".format(file.size / (1024 * 1024))}
        st.write(file_details)
        
        blob = cv2.dnn.blobFromImage(img2, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i-1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs,img2)
        
        st.image(img2, caption='Proccesed Image.')
        
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        my_bar.progress(100)


object_detection_image()