import streamlit as st
import awesome_streamlit as ast
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageOps
from img_classification import teachable_machine_classification
import joblib
import matplotlib.pyplot as plt




# Load the data
data = pd.read_csv(r"C:\Users\jeffe\OneDrive\Desktop\MOM\data.csv")

# Define the app title
title_text = "<span style='color: hotpink; font-size: 70px;'>L<span style='color: white; background-color: hotpink; border-radius: 5px; padding: 0px 5px;'>AID</span></span>"

# Set page configuration
st.set_page_config(page_title="LYDIA ARTIFICIAL INTELLIGENCE DIAGNOSTICS", page_icon=":guardsman:", layout="wide")

# Add the title to the page
st.markdown(title_text, unsafe_allow_html=True)

# Add content to the page below the title
st.write("<h2 style='color: gray;'>LYDIA ARTIFICIAL INTELIGENCE DIAGNOSTICS</h2>", unsafe_allow_html=True)
st.write("LYDIA ARTIFICIAL INTELLIGENCE DIAGNOSTICS is your reliable and accurate tool for breast cancer diagnosis. Our app utilizes state-of-the-art algorithms and machine learning techniques to analyze medical data and provide personalized recommendations for each patient. With our user-friendly interface, you can input your medical data and receive a detailed diagnosis report in no time. Whether you're a healthcare provider or a patient, LYDIA ARTIFICIAL INTELLIGENCE DIAGNOSTICS is the ideal tool for timely treatment and improved outcomes. Try out our app today and experience the power of cutting-edge technology in healthcare!")



# Add the title to the sidebar
st.sidebar.markdown(title_text, unsafe_allow_html=True)
st.sidebar.title('MODELS')
model_type = st.sidebar.selectbox('Choose the model type', ('Convolutional Neural Networks(CNNs)','Logistic Regression', 'Random Forest'))
st.sidebar.markdown('---')





# Define a function to display the help content
def show_help():
    st.write("### Help Section")
    st.write("This app uses machine learning to diagnose breast cancer. To use the app, follow these steps:")
    st.write("1. Enter the required information in the sidebar.")
    st.write("2. Choose among the AI model you would like to use.")
    st.write("3. Review the diagnosis and consult with the medical professionals.")

# Create a beta_expander widget for the help section
with st.expander("ℹ️ Help"):
    show_help()
# Load the sample data from a CSV file
data = pd.read_csv(r"C:\Users\jeffe\OneDrive\Desktop\MOM\data.csv")

# Show the sample data in a table
st.write("Sample of the Dataset that was used for training:")
st.write(data.head(5))

st.header("Breast Cancer Risk Calculator")

with st.sidebar:
    st.title("Enter Your Information ")
    st.subheader("Breast Cancer Risk Calculator")
    
    # Gender
    gender = st.radio("Gender", options=["Male", "Female"])
    
    # Age
    age = st.slider("Age", 18, 100)
    
    # Family history
    family_history = st.radio("Family History of Breast Cancer?", options=["Yes", "No"])
    
    # Genetic mutations
    genetic_mutations = st.radio("Have you been tested for BRCA1 or BRCA2 gene mutations?", options=["Yes", "No"])
    
    # Personal history
    personal_history = st.radio("Have you had breast cancer in the past?", options=["Yes", "No"])
    
    # Breast changes
    breast_changes = st.radio("Have you had any atypical breast changes or conditions?", options=["Yes", "No"])
    
    # Radiation therapy
    radiation_therapy = st.radio("Have you had radiation therapy to the chest area before the age of 30?", options=["Yes", "No"])
    
    # Hormone therapy
    hormone_therapy = st.radio("Have you used hormone therapy for menopause symptoms?", options=["Yes", "No"])

risk_score = 0

# Assign a value to each risk factor
if gender == "Female":
    risk_score += 1
if age >= 55:
    risk_score += 1
if family_history == "Yes":
    risk_score += 1
if genetic_mutations == "Yes":
    risk_score += 2
if personal_history == "Yes":
    risk_score += 1
if breast_changes == "Yes":
    risk_score += 1
if radiation_therapy == "Yes":
    risk_score += 1
if hormone_therapy == "Yes":
    risk_score += 1

# Calculate the risk percentage
risk_percentage = risk_score * 10

# Define the colors for the pie chart
colors=["hotpink", "skyblue"]

# Create a pie chart with values and 
fig, ax = plt.subplots(figsize=(1,1))
fig, ax = plt.subplots()
ax.pie([risk_percentage, 100 - risk_percentage], labels=["Risk Percentage", ""], colors=colors, autopct='%.1f%%')
ax.axis("equal")  # Equal aspect ratio ensures that the pie is drawn as a circle.
st.pyplot(fig)


st.write("NOTE: Thank you for using our breast cancer risk calculator. We want to remind you that the calculated risk is not the end result, but rather an estimation based on the information provided. It is important to understand that this calculator is not a definitive diagnosis of breast cancer. We recommend consulting with a healthcare provider to discuss your individual risk factors and the appropriate screening schedule. Regular screenings and early detection are key in the fight against breast cancer.")


st.sidebar.title('About')
st.sidebar.info('This app was created by [Mulamba Joram Jefferson](jeffersonjoram@gmail.com)')
st.sidebar.info('[Message Joram on WhatsApp & For inquiries contact +256709863018](https://wa.me/message/AWYFSSTTGLENJ1)')   

from PIL import Image
import io

# Define function to convert PGM image to PNG
def pgm_to_png(pgm_data):
    pgm_bytes = pgm_data.read()
    pgm_file = io.BytesIO(pgm_bytes)
    with Image.open(pgm_file) as img:
        # Convert image to numpy array and convert pixel values to uint8
        img_arr = np.array(img)
        img_arr = img_arr.astype('uint8')

        # Create new PIL Image object from numpy array and save as PNG
        png_image = Image.fromarray(img_arr)
        png_bytes = io.BytesIO()
        png_image.save(png_bytes, format='png')
        return png_bytes.getvalue()

# Define Streamlit app

# Prompt user to upload PGM image
st.header("Convert your DCOM and PGM Images from Here and Download")
pgm_data = st.file_uploader("Upload PGM image", type=["pgm"], key="dcom")

# If user has uploaded a PGM image, convert it to PNG and download
if pgm_data is not None:
    png_bytes = pgm_to_png(pgm_data)
    st.image(png_bytes, caption="Converted PNG Image", width=400)
    st.download_button(
        label="Download PNG Image",
        data=png_bytes,
        file_name="converted_image.png",
        mime="image/png"
    )



import pydicom

def dicom_to_png(dicom_data):
    dicom_bytes = dicom_data.read()
    dicom_file = io.BytesIO(dicom_bytes)
    dicom_image = pydicom.dcmread(dicom_file)
    png_image = dicom_image.pixel_array
    png_bytes = io.BytesIO()
    png_image = Image.fromarray(png_image.astype('uint16'), mode='L')
    png_image.save(png_bytes, format='png')
    return png_bytes.getvalue()


uploaded_file = st.file_uploader("Upload DICOM Image", type="dcm", key="DCOM")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded DICOM Image", width=400)
    if st.button("Convert to PNG"):
        png_bytes = dicom_to_png(uploaded_file)
        st.image(png_bytes, caption="Converted PNG Image", width=400)
        st.download_button(
            label="Download PNG Image",
            data=png_bytes,
            file_name="converted_image.png",
            mime="image/png"
        )




st.header("Breast Cancer Mammogram Diagnosis")
st.text("Upload a scan for Diagnosis")
model3 = tf.keras.models.load_model(r"C:\Users\jeffe\OneDrive\Desktop\MOM\keras_model3.h5")

uploaded_file3 = st.file_uploader("Choose a scan ...", type="png", key="file3")
if uploaded_file3 is not None:
    image = Image.open(uploaded_file3)
    st.image(image, caption='Uploaded Scan.', use_column_width=True)
    st.write("")
    st.write("DIAGNOSING...")
    label = teachable_machine_classification(image, r"C:\Users\jeffe\OneDrive\Desktop\MOM\keras_model3.h5")
    model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if label == 0:
        st.write("The scan is MALIGNANT")
    elif label == 1:
        st.write("The scan is BENIGN")
    else:
        st.write("UNRECOGNISED(Consult with a Medical Doctor)")



st.header("Breast Cancer Ultrasound Diagnosis")
st.text("Upload a scan for Diagnosis")
model = tf.keras.models.load_model(r"C:\Users\jeffe\OneDrive\Desktop\MOM\keras_model.h5")

uploaded_file1 = st.file_uploader("Choose a scan ...", type="png", key="file1")
if uploaded_file1 is not None:
    image = Image.open(uploaded_file1)
    st.image(image, caption='Uploaded Scan.', use_column_width=True)
    st.write("")
    st.write("DIAGNOSING......")
    label = teachable_machine_classification(image, r"C:\Users\jeffe\OneDrive\Desktop\MOM\keras_model.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if label == 0:
        st.write("The scan is NORMAL")
    elif label == 1:
        st.write("The scan is MALIGNANT")
    else:
        st.write("The scan is BENIGN")


st.header("Breast Cancer Histopathology Image Diagnosis")
st.text("Upload a scan for Diagnosis")
model2 = tf.keras.models.load_model(r"C:\Users\jeffe\OneDrive\Desktop\MOM\keras_model2.h5")

uploaded_file2 = st.file_uploader("Choose a scan ...", type="png", key="file2")
if uploaded_file2 is not None:
    image = Image.open(uploaded_file2)
    st.image(image, caption='Uploaded Scan.', use_column_width=True)
    st.write("")
    st.write("DIAGNOSING...")
    label = teachable_machine_classification(image, r"C:\Users\jeffe\OneDrive\Desktop\MOM\keras_model2.h5")
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if label == 0:
        st.write("The scan is MALIGNANT")
    elif label == 1:
        st.write("The scan is BENIGN")
    else:
        st.write("UNRECOGNISED(Consult with a Medical Doctor)")
       

            
# Define the feedback and support expanders
with st.expander("Feedback"):
    st.write("Please provide your feedback below:")
    feedback = st.text_input("Feedback", "")
    if st.button("Submit Feedback"):
        # Send feedback to a database or email address
        st.success("Thank you for your feedback!")
               
with st.expander("Support"):
    st.write("If you need help, please contact us at jeffersonjoram@gmail.com.")






import random
import uuid


# Define the responses for each question
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey!"],
    "help": ["How can I help you?", "What do you need help with?"],
    "breast_cancer": ["Breast cancer is a type of cancer that starts in the breast tissue.", "Breast cancer is a disease in which malignant (cancer) cells form in the tissues of the breast."],
    "risk_factors": ["Women over the age of 50 are at higher risk of breast cancer.", "Women who have a family history of breast cancer are at higher risk.", "Women who have certain gene mutations, such as BRCA1 or BRCA2, are at higher risk."],
    "check": ["Breast self-exams and mammograms are two common ways to check for breast cancer."],
    "prevent": ["Some ways to help prevent breast cancer include maintaining a healthy weight, exercising regularly, limiting alcohol consumption, and not smoking."],
    "treatment": ["If you are diagnosed with breast cancer, your doctor will recommend a treatment plan based on your specific case. This may include surgery, radiation therapy, chemotherapy, or a combination of these treatments."],
    "farewell": ["Goodbye!", "Take care!", "See you later!"]
}

# Define a function to generate a response to the user's input
def get_response(user_input):
    user_input = user_input.lower()

    if "hello" in user_input or "hi" in user_input:
        return random.choice(responses["greeting"])
    
    elif "help" in user_input or "what can you do" in user_input:
        return random.choice(responses["help"])
    
    elif "breast cancer" in user_input or "what is breast cancer" in user_input:
        return random.choice(responses["breast_cancer"])
    
    elif "at risk" in user_input or "who is at risk" in user_input:
        return random.choice(responses["risk_factors"])
    
    elif "check" in user_input or "how to check" in user_input:
        return random.choice(responses["check"])
    
    elif "prevent" in user_input or "how to prevent" in user_input:
        return random.choice(responses["prevent"])
    
    elif "treatment" in user_input or "what to do if diagnosed" in user_input:
        return random.choice(responses["treatment"])
    
    elif "bye" in user_input or "goodbye" in user_input:
        return random.choice(responses["farewell"])
    
    else:
        return "I'm sorry, I don't understand. Can you please rephrase your question?"

# Define a function to handle the chat with the user
st.header("Welcome to the Breast Cancer Chatbot!")
st.write("Please enter your questions or type 'bye' to exit.")



user_input = st.text_input("You:")
        
st.write("LYDIAbot:", get_response(user_input))


# Add some additional information
st.header("Additional Information")
st.write("Breast cancer is the second most common cancer in women worldwide, and the most common cancer in women in developed countries. Early detection and treatment can greatly improve the chances of survival. This AI diagnostic tool is intended to be used for informational purposes only and should not replace the advice of a medical professional. Please consult your doctor if you have any concerns about your breast health.")
st.write('App created by Mulamba Joram Jefferson(jeffersonjoram@gmail.com)')