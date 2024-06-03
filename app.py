import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image as img_preprocessing
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="Pupillometry Analysis System", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

selected_model = "EfficientNetB0"
model = keras.models.load_model("models\\EfficientNetB0_model.h5")


def predict(image_path, model):
    img = img_preprocessing.load_img(image_path, target_size=(224, 224))
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    classes = ['Age Degeneration', 'Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Normal', 'Others']
    return classes[np.argmax(predictions)]

def generate_medical_report(predicted_label):
    # Define class labels and corresponding medical information
    medical_info = {
        "Age Degeneration": {
            "report": "The patient appears to have age-related degeneration. Further evaluation and management are recommended to prevent vision loss.",
            "preventative_measures": [
                "Regular eye exams are crucial for early detection and intervention",
                "Maintain a healthy lifestyle with a balanced diet and regular exercise",
                "Protect eyes from UV rays with sunglasses when outdoors",
            ],
            "precautionary_measures": [
                "Schedule regular follow-ups with an eye specialist",
                "Consider supplements recommended by your doctor to support eye health",
            ],
        },
        "Cataract": {
            "report": "It seems like the patient has cataracts. While common and treatable, it's important to address symptoms and consider treatment options.",
            "preventative_measures": [
                "Protect eyes from UV exposure with sunglasses",
                "Quit smoking if applicable, as it can increase cataract risk",
                "Maintain overall health with a balanced diet and regular exercise",
            ],
            "precautionary_measures": [
                "Consult with an eye specialist for personalized treatment options",
                "Discuss surgical options if cataracts significantly affect daily activities",
            ],
        },
        "Diabetes": {
            "report": "The patient appears to have diabetes. It's crucial to manage blood sugar levels effectively to prevent complications, including diabetic retinopathy.",
            "preventative_measures": [
                "Monitor blood sugar levels regularly as advised by your doctor",
                "Follow a diabetic-friendly diet rich in fruits, vegetables, and whole grains",
                "Engage in regular physical activity to improve insulin sensitivity",
            ],
            "precautionary_measures": [
                "Attend regular check-ups with healthcare providers to monitor diabetes management",
                "Consult with an ophthalmologist to assess eye health and discuss preventive measures",
            ],
        },
        "Glaucoma": {
            "report": "The patient shows signs of glaucoma. Early detection and treatment are essential to prevent vision loss.",
            "preventative_measures": [
                "Attend regular eye exams, especially if at risk for glaucoma",
                "Follow treatment plans prescribed by your eye specialist",
                "Manage intraocular pressure through medication or other interventions",
            ],
            "precautionary_measures": [
                "Be vigilant for changes in vision and report them promptly to your doctor",
                "Discuss surgical options if medication alone isn't controlling glaucoma effectively",
            ],
        },
        "Hypertension": {
            "report": "It appears the patient has hypertension. Proper management is crucial to prevent potential eye complications.",
            "preventative_measures": [
                "Monitor blood pressure regularly and follow treatment plans prescribed by your doctor",
                "Adopt a heart-healthy diet low in sodium and high in fruits and vegetables",
                "Engage in regular physical activity to help lower blood pressure",
            ],
            "precautionary_measures": [
                "Attend regular check-ups with healthcare providers to monitor blood pressure control",
                "Inform your eye specialist about hypertension diagnosis for comprehensive care",
            ],
        },
        "Myopia": {
            "report": "The patient appears to have myopia. While common, it's important to monitor vision changes and consider corrective measures if needed.",
            "preventative_measures": [
                "Attend regular eye exams to monitor vision changes",
                "Take breaks during prolonged near work to reduce eye strain",
                "Consider corrective lenses or refractive surgery if vision significantly affects daily activities",
            ],
            "precautionary_measures": [
                "Discuss with an eye specialist for personalized recommendations based on severity",
                "Monitor for any progression of myopia and adjust treatment as necessary",
            ],
        },
        "Normal": {
            "report": "Great news! It seems like the patient's eyes are healthy. Regular check-ups are recommended to maintain eye health.",
            "preventative_measures": [
                "Continue with regular eye exams for ongoing monitoring",
                "Maintain overall health with a balanced diet and regular exercise",
                "Protect eyes from UV exposure with sunglasses when outdoors",
            ],
            "precautionary_measures": [
                "Stay informed about any changes in vision and report them promptly",
                "Schedule annual comprehensive eye check-ups to ensure continued eye health",
            ],
        },
        "Others": {
            "report": "The patient's condition falls into a category not specifically listed. Further evaluation and consultation with a healthcare provider are recommended.",
            "preventative_measures": [
                "Attend follow-up appointments as advised by your healthcare provider",
                "Discuss any concerns or symptoms with your doctor for appropriate management",
                "Follow recommended lifestyle measures for overall health and well-being",
            ],
            "precautionary_measures": [
                "Seek clarification from your healthcare provider regarding your specific condition",
                "Follow treatment plans or recommendations provided by specialists involved in your care",
            ],
        },
    }

    # Retrieve medical information based on predicted label
    medical_report = medical_info[predicted_label]["report"]
    preventative_measures = medical_info[predicted_label]["preventative_measures"]
    precautionary_measures = medical_info[predicted_label]["precautionary_measures"]

    # Generate conversational medical report with each section in a paragraphic fashion
    report = (
        "Medical Report:\n\n"
        + medical_report
        + "\n\nPreventative Measures:\n\n- "
        + ",\n- ".join(preventative_measures)
        + "\n\nPrecautionary Measures:\n\n- "
        + ",\n- ".join(precautionary_measures)
    )

    precautions = precautionary_measures

    return report, precautions




# Apply custom CSS for aesthetics

def main():
    st.markdown(
    """
    <style>
        body {
            background-color: #0b1e34;
            color: white;
        }
        .st-bw {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)


    st.markdown(
        """
        <style>
            .centered-image {
                display: flex;
                justify-content: center;
            }
            .centered-image img {
                width: 90%;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Pupillometry Analysis System")
    # Model selection
    model_options = ["VGG16"]
    model = keras.models.load_model("models\\EfficientNetB0_model.h5")
    selected_model = st.selectbox("Select a model:", model_options)
    if st.button("Load Model"):
        if selected_model == "EfficientNetB0":
            model = keras.models.load_model("models\\EfficientNetB0_model.h5")
        elif selected_model == "VGG16":
            model = keras.models.load_model("models\\VGG16_model.h5")
        elif selected_model == "VGG19":
            model = keras.models.load_model("models\\VGG19_model.h5")
        elif selected_model == "DenseNet169":
            model = keras.models.load_model("models\\DenseNet169_model.h5")
        elif selected_model == "ResNet50":
            model = keras.models.load_model("models\\ResNet50_model.h5")
        elif selected_model == "Xception":
            model = keras.models.load_model("models\\Xception_model.h5")
        elif selected_model == "InceptionV3":
            model = keras.models.load_model("models\\InceptionV3_model.h5")
            
        st.success(f"Model {selected_model} has been loaded successfully.")

    # File uploader
    st.title("Upload Pupil Image")
    uploaded_image = st.file_uploader(
                "Choose a Pupil image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
            )
    if uploaded_image is not None:

    # Display the image within a centered container
        st.markdown("<div class='centered-image'>", unsafe_allow_html=True)
        st.image(uploaded_image, caption='Uploaded Image')
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Predict Condition"):
            condition = predict(uploaded_image, model)
            st.write("Predicted Condition: ", condition)
            report, precautions = generate_medical_report(condition)
            st.write(report)
            st.write("\nAdditional Precautionary Measures:\n- " + ",\n- ".join(precautions))
if __name__ == "__main__":
    main()
