import streamlit as st
import plotly.express as px
import numpy as np
import math
from PIL import Image
import pickle
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import keras
import tensorflow
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


st.title("Image Classification 👀")


#import model
model1 = tensorflow.keras.models.load_model('final_model.h5')

#import file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg",'tif'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image 📸', use_column_width=True)
    st.write("")
    st.write("Classifying... ⏳")

#predict
try: 
    #use keras function to convert image to numpy array
    img_array = img_to_array(image)
    #st.text(f"Image array shape: {img_array.shape}")
    #add dimension for number of images
    img_array = np.expand_dims(img_array, axis = 0)
    #st.text(f"Image array shape (after): {img_array.shape}")
    p_model = model1.predict(img_array)
    #pred = (p_model > 0.5).astype("int32")
    #st.text(f'prediction is:{p_model}')
except:
    st.subheader('Go ahead. Were waiting...')
    st.stop()

#return prediction
if p_model[0][0] == math.isclose(1,1.0,rel_tol = .0000001 ):
    pred_text = "Building 🏘"
    st.subheader(f"We predict {pred_text}")
elif p_model[0][1] == math.isclose(1,1.0,rel_tol = .0000001 ):
    pred_text = "Forest 🌲"
    st.subheader(f"We predict {pred_text}")
elif p_model[0][2] == math.isclose(1,1.0,rel_tol = .0000001 ):
    pred_text = "Glacier 🗻"
    st.subheader(f"We predict {pred_text}")
elif p_model[0][3] == math.isclose(1,1.0,rel_tol = .0000001 ):
    pred_text = "Mountain ⛰"
    st.subheader(f"We predict {pred_text}")
elif p_model[0][4] == math.isclose(1,1.0,rel_tol = .0000001 ):
    pred_text = "Sea 🌅"
    st.subheader(f"We predict {pred_text}")
else:
    pred_text = "Street 🚦"
    st.subheader(f"We predict {pred_text}")
st.balloons()
