import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

import re
import time
import joblib
import contractions

from PIL import Image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import vgg16, resnet50


MAX_SEQUENCE_LENGTH = 22
IMG_TARGET_SIZE = (224, 224)


# Function to clean the questions
def clean_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    return text
    

def encode_questions_model_1(tknizr, embd_layer, text, seq_len):
    # clean the text
    text = clean_text(text)
    # converting to int sequences using the tokenizer
    encoded_seq = tknizr.texts_to_sequences([text]) 
    # padding sequences to seq_len
    encoded_seq = pad_sequences(
        encoded_seq, maxlen=seq_len, dtype='int32', padding='post')
    # embed using the Embedding layer
    encoded_seq = embd_layer(encoded_seq)
    return encoded_seq


def preprocess_images_model_1(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_arr = image.img_to_array(img)
    img_arr = vgg16.preprocess_input(img_arr)
    return img_arr


def encode_images_model_1(vgg_featurizer, img_path, target_size):
    # preprocess the image
    img_arr = preprocess_images_model_1(img_path, target_size)
    # vgg_featurizer
    vgg_feats = vgg_featurizer(np.expand_dims(img_arr, axis=0))
    return vgg_feats


def produce_ans_model_1(img_file, question):
    # encode the questions using the "encode_questions_model_1" funtion defined earlier
    enc_seq = encode_questions_model_1(
        data['tknizr'], data['glove_embedding_layer'], question, MAX_SEQUENCE_LENGTH)[0]
    # encode the images using the "encode_images_model_1" funtion defined earlier
    vgg_feats = encode_images_model_1(
        data['vgg_featurizer'], img_file, IMG_TARGET_SIZE)[0]
    # generate answer
    y_pred_val = data['model_1']([np.asarray([enc_seq]), np.asarray([vgg_feats])])
    y_pred_val = data['ohe'].inverse_transform(tf.one_hot(tf.argmax(y_pred_val, axis=1), depth=len(data['ohe'].categories_[0])))[0][0]
    return y_pred_val


html1="""
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h1>Visual Question Answering </h1>
      <h1>(VQA)</h1>
    </div>
      """
st.markdown(html1,unsafe_allow_html=True) #simple html 

@st.cache(allow_output_mutation=True)
def load_model_and_featurizers():
    data = {}

    data['ohe'] = joblib.load("featurizers/ohe.joblib")
    data['model_1'] = load_model('model_save/best_model_1.h5')

    # Tokenizer
    data["tknizr"] = joblib.load("featurizers/tknizr.joblib")
    # indices for each of the word in the vocab
    WORD_INDEX = data["tknizr"].word_index
    # vocab size (no. of all tokens in the text(questions) train data)
    VOCAB_SIZE = len(WORD_INDEX) + 1 # +1 because all the indices we get are starting from 1 but after padding we also add 0 so total=len(word_index)+1

    EMBEDDING_DIM = 300

    embeddings_index = dict()
    f = open('featurizers/glove.6B.300d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for word, i in WORD_INDEX.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Embedding layer
    data['glove_embedding_layer'] = Embedding(
        VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], 
        input_length=MAX_SEQUENCE_LENGTH, name='GLOVE_embedding_layer', 
        trainable=False
    )

    # ==============================================================================
    # VGG16 Featurizer. 
    # The weights obtained are a result of training on 'imagenet' dataset. 
    vgg16_model = vgg16.VGG16(
        weights='imagenet', 
        input_shape=IMG_TARGET_SIZE+ (3,), #`input_shape` must be a tuple of three integers. The input must have 3 channels if weights = 'imagenet'
    )
    vgg16_model.trainable = False

    # Create a model that outputs activations from vgg16's last hidden layer
    data['vgg_featurizer'] = Model(
        inputs=vgg16_model.inputs, outputs=vgg16_model.layers[-2].output)
    data['vgg_featurizer'].trainable = False
    return data

# ------------------------------------- APP -----------------------------------

with st.sidebar.form("Info"):
    st.title("Hey Whatssup!")
    Name = st.text_input("Full Name")
    Contact_Number = st.text_input("Contact Number")
    Email_address = st.text_input("Email address")

    if not Name and Email_address:
        st.warning("Please fill out your Name and Email-ID")

    if Name and Contact_Number and Email_address:
        st.success("Thanks!")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Name:", Name, "Email:", Email_address)

# Create a text element and let the reader know the data is loading.
# data_load_state = st.spinner("Loading required files...")
# data = load_model_and_featurizers()
# Notify the reader that the data was successfully loaded.
placeholder = st.empty()
with placeholder.container():
    # with st.spinner('Loading required files...'):
    data = load_model_and_featurizers()
    if data:
        st.info('Loading requied files...done!', icon="ℹ️")
        time.sleep(2)
# Clear all those elements:
placeholder.empty()

col1, col2 = st.columns(2, gap="large")

with col1:
    tab1, tab2= st.tabs(["Upload", "Capture"])
    with tab1:
        st.subheader("Choose a file to upload: ")
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png"])
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, use_column_width='always')

    with tab2:
        click = st.button("Open camera")
        if click==True:
            uploaded_file = st.camera_input("Click a picture")


with col2:
    eng, hin= st.tabs(["English","Hindi"])

    with eng:
        st.subheader("Question: ")
        question = st.text_area('Ask a Question').strip()

        submit_btn = st.button('Submit')
        if submit_btn:
            # Progress bar
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.03)
                my_bar.progress(percent_complete + 1)
            # Snow
            st.snow()
            ans = produce_ans_model_1(uploaded_file, question)
            #result will be displayed if button is pressed
            st.success("Your answer is : "
                    "{}".format(ans))
        else:
            ans = ""

        with hin:
            st.write("Stay tuned to experience the feature🙂")

st.subheader("Here are some nerdy analytics 😁")

# Expander
with st.expander("See explanation"):
    st.write("Plot")

html2="""
<div style="text-shadow: 3px 1px 2px purple; color:white; margin:80px; text-align:center;font-size: large;">
    <span> Developed by Group 12</span>
</div>
    """

st.markdown(html2,unsafe_allow_html=True)