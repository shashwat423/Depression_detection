import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load trained model, scaler, and accuracy
model = joblib.load('depression_model.pkl')  
scaler = joblib.load('scaler.pkl')  
accuracy = joblib.load('model_accuracy.pkl')  

# Load dataset to extract feature names
dataset_path = "Depression Professional Dataset.csv"
df = pd.read_csv(dataset_path)

# Identify feature columns (excluding target 'Depression')
feature_columns = [col for col in df.columns if col.lower() != "depression"]

# Define categorical features for dropdown selection
categorical_mappings = {
    "Gender": ["Male", "Female", "Other"],
    "Job Satisfaction": ["Low", "Medium", "High"],
    "Dietary Habits": ["Healthy", "Average", "Unhealthy"],
    "Have you ever had suicidal thoughts ?": ["Yes", "No"],
    "Family History of Mental Illness": ["Yes", "No"]
}

# Streamlit UI
st.title("🧠 Depression Detection System")
st.write("Enter your details below to check if you may be experiencing depression.")

# Sidebar to show model accuracy
st.sidebar.subheader("📊 Model Performance")
st.sidebar.write(f"**Accuracy: {accuracy * 100:.2f}%**")

# User input fields
user_data = []
encoded_data = []

for feature in feature_columns:
    if feature in categorical_mappings:
        value = st.selectbox(feature, categorical_mappings[feature])
        encoded_value = categorical_mappings[feature].index(value)  # Convert to numerical
    else:
        value = st.number_input(feature, min_value=0.0, max_value=100.0, step=0.1)
        encoded_value = value  # Keep numerical input as is

    user_data.append(value)
    encoded_data.append(encoded_value)

# Predict depression when button is clicked
if st.button("Predict"):
    try:
        input_data = np.array(encoded_data).reshape(1, -1)
        input_data = scaler.transform(input_data)  # Normalize input
        prediction = model.predict(input_data)[0]
        prediction_prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ You may be experiencing depression. Probability: {prediction_prob:.2f}")
        else:
            st.success(f"✅ No signs of depression detected. Probability: {prediction_prob:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

# ROC Curve Display
st.subheader("📈 ROC Curve")
try:
    y_test = joblib.load('y_test.pkl')  # Load actual test labels
    y_prob = joblib.load('y_prob.pkl')  # Load predicted probabilities

    # Compute ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')

    # Display Plot
    st.pyplot(fig)
except FileNotFoundError:
    st.warning("🚨 ROC Curve data missing! Ensure 'y_test.pkl' and 'y_prob.pkl' exist.")
