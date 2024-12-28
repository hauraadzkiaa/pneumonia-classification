import streamlit as st  
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Set page configuration
st.set_page_config(
    page_title="Pneumonia and COVID Classification",
    page_icon=":mask:",
    initial_sidebar_state='auto'
)

# Hide Streamlit default elements (optional)
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load your trained model
model = tf.keras.models.load_model('/Users/hauradelfina/Documents/haura adzkia/project/image classification/result/GoogLeNet_model.h5')  # Update with your model's path

# Define the class names
class_names = ['Healthy', 'Pneumonia', 'COVID-19']

# Sidebar
with st.sidebar:
    # st.image('path_to_logo.png', use_column_width=True)  # Update with your logo's path
    st.title("Pneumo-COVID Classifier")
    st.subheader("Accurate detection of pneumonia and COVID-19 through chest X-rays.")

# Main Page
st.write("""
# Pneumonia and COVID Classification
Upload a chest X-ray image to determine if the subject has Pneumonia, COVID-19, or is Healthy.
""")

# File uploader
file = st.file_uploader("", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    """Preprocess and predict the class of the input image."""
    size = (224, 224)  # Resize to match the model input size
    # image = ImageOps.fit(image_data, size, Image.ANTIALIAS) 
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)

    image = image.convert("RGB")

    img_array = np.asarray(image) / 255.0  # Normalize the image
    img_reshape = img_array[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction


# Classification logic
if file is None:
    st.text("Please upload an X-ray image file.")
else:
    image = Image.open(file)
    # st.image(image, use_column_width=True)
    st.image(image, use_container_width=True)
    predictions = import_and_predict(image, model)

    # Display results
    detected_class = class_names[np.argmax(predictions)]
    st.sidebar.markdown(f"### Detected Class: **{detected_class}**")
    
    # Provide recommendations based on the detected class
    if detected_class == 'Healthy':
        st.balloons()
        st.success("The subject is healthy. No issues detected.")
    elif detected_class == 'Pneumonia':
        st.warning("Pneumonia detected.")
        st.markdown("### Suggested Actions:")
        st.info("""
        - Consult a healthcare provider immediately.
        - Administer prescribed antibiotics or antivirals as advised.
        - Ensure adequate hydration and rest.
        """)
    elif detected_class == 'COVID-19':
        st.error("COVID-19 detected.")
        st.markdown("### Suggested Actions:")
        st.info("""
        - Isolate the patient to prevent further spread.
        - Seek immediate medical attention.
        - Monitor oxygen levels and consult a doctor for antivirals or other treatment.
        """)

    # Optional: Add accuracy details
    st.sidebar.info(f"Model Confidence: {np.max(predictions) * 100:.2f}%")
