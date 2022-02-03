# Imports
from PIL import Image
import streamlit as st
import tensorflow as tf
import time
import os

class_names = ["Cat", "Dog"]

def get_predictions(input_image):
    tflite_interpreter = tf.lite.Interpreter(model_path="model_fp16.tflite")
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    tflite_interpreter.allocate_tensors()
    tflite_interpreter.set_tensor(input_details[0]["index"], input_image)
    tflite_interpreter.invoke()
    tflite_model_prediction = tflite_interpreter.get_tensor(output_details[0]["index"])
    tflite_model_prediction = tflite_model_prediction.squeeze().argmax(axis = 0)
    pred_class = class_names[tflite_model_prediction]
    print(tflite_model_prediction)
    print(pred_class)
    #score = tf.nn.softmax(tflite_model_prediction)
    #print(score)
    #print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
    #inference = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    #print(inference)
    return pred_class 

## Page Title
st.set_page_config(page_title = "Cats vs Dogs Image Classification")
st.title(" Cat vs Dogs Image Classification")
st.markdown("---")

## Sidebar
#st.sidebar.header(" Cat vs Dogs Image Classification")



st.header(" Cat vs Dogs Image Classification")

## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png", 'jpeg'])
if uploaded_file is not None:
    with open(os.path.join("tempDir",uploaded_file.name),"wb") as f:
         f.write(uploaded_file.getbuffer())
    path = os.path.join("tempDir",uploaded_file.name)
    img = tf.keras.preprocessing.image.load_img(path , grayscale=False, color_mode='rgb', target_size=(300,300,3), interpolation='nearest')
    st.image(img)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

if st.button("Get Predictions"):
    suggestion = get_predictions(input_image =img_array)
    st.success(suggestion)
