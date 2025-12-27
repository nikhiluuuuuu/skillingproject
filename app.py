import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config("AI Pneumonia Detection", "🫁", layout="centered")

st.markdown("""
<style>
.result {padding:20px;border-radius:10px;font-size:20px;text-align:center}
.ok {background:#d4edda;color:#155724}
.bad {background:#f8d7da;color:#721c24}
</style>
""", unsafe_allow_html=True)

st.title("🫁 AI-Powered Pneumonia Detection")
st.caption("Chest X-ray analysis using Deep Learning")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mymodel.h5")

model = load_model()

file = st.file_uploader("Upload Chest X-ray (JPG / PNG)", ["jpg","jpeg","png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_column_width=True)

    if st.button("Analyze X-ray"):
        img_resized = img.resize((224,224))
        arr = image.img_to_array(img_resized)/255.0
        arr = np.expand_dims(arr,0)

        prob = model.predict(arr)[0][0]
        label = "PNEUMONIA" if prob > 0.5 else "NORMAL"

        st.subheader("Diagnosis Result")
        st.markdown(
            f"<div class='result {'bad' if label=='PNEUMONIA' else 'ok'}'>"
            f"{label}<br>Confidence: {prob:.2f}</div>",
            unsafe_allow_html=True
        )

st.warning("⚠️ Educational use only. Not a medical diagnosis tool.")