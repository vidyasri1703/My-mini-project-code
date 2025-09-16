from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
import argparse
import io
import os
from PIL import Image
from flask import Flask, render_template, request, redirect, Response

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_path1 = 'model.h5' # load .h5 Model


model = load_model(model_path1, compile=False)


from tensorflow.keras.preprocessing.image import load_img, img_to_array

def model_predict1(image_path,model):
    print("Predicted")
    image = load_img(image_path,target_size=(128,128))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))
    print(result)
   
    if result == 0:
        return "LAND-SLIDE DETECTED!","result.html"        
    elif result == 1:
        return "NO LAND-SLIDE DETECTED!","result.html"
    
    
@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    print("Entered")
    
    print("Entered here")
    file = request.files['file'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
        
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    print("@@ Predicting class......")
    pred, output_page = model_predict1(file_path,model)
             
    return render_template(output_page, pred_output = pred, img_src=UPLOAD_FOLDER + file.filename)

@app.route('/notebook')
def notebook():
	return render_template('Notebook.html')
   
if __name__ == '__main__':
    app.run(debug=False)