import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image
import pickle

# ---- Load the model and tokenizer ----
model = load_model('Inception_final.keras')  # Path to your trained CNN+LSTM model
with open('tokenizer_final.pkl', 'rb') as handle:  # Path to your saved tokenizer
    tokenizer = pickle.load(handle)

max_length = 38  # Set according to your training
vocab_size = len(tokenizer.word_index) + 1  # vocab_size used during training

# ---- Define your feature extractor ----
# (Rebuild the same InceptionV3 / Xception / ResNet feature model you used)
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

base_model = InceptionV3(weights='imagenet')
cnn_model = Model(base_model.input, base_model.layers[-2].output)

def extract_features(img):
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = cnn_model.predict(img_array, verbose=0)
    return feature

# ---- Caption Generation Logic ----
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()[1:-1]  # remove startseq and endseq
    return ' '.join(final_caption)

# ---- Streamlit UI ----
st.title("🖼️ Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
captured_image = st.camera_input("Or take a photo")

img = None
if uploaded_file is not None:
    img = Image.open(uploaded_file)
elif captured_image is not None:
    img = Image.open(captured_image)

if img is not None:
    st.image(img, caption='Uploaded or Captured Image', use_column_width=True)
    
    with st.spinner('Generating caption...'):
        photo_feature = extract_features(img)
        caption = generate_caption(model, tokenizer, photo_feature, max_length)
    
    st.markdown("### 📜 Generated Caption:")
    st.success(caption)