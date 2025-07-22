import streamlit as st
import pandas as pd
import joblib

# Load Model and Encoders
model = joblib.load('bank_model.sav')
country_encoder = joblib.load('country_encoder.sav')
location_encoder = joblib.load('location_type_encoder.sav')
gender_encoder = joblib.load('gender_of_respondent_encoder.sav')
relationship_encoder = joblib.load('relationship_with_head_encoder.sav')
marital_encoder = joblib.load('marital_status_encoder.sav')
education_encoder = joblib.load('education_level_encoder.sav')
job_encoder = joblib.load('job_type_encoder.sav')
cellphone_encoder = joblib.load('cellphone_access_encoder.sav')
model_columns = joblib.load('model_columns.sav')  # List of expected model input columns

# App Title 
st.title("Financial Inclusion Prediction")
st.subheader("Will this person have a bank account?")

#  Sidebar Input
with st.sidebar:
    st.header(" Respondent Info")

    country = st.selectbox("Country", country_encoder.classes_)
    year = st.selectbox("Year", [2016, 2017, 2018])
    location_type = st.selectbox("Location Type", location_encoder.classes_)
    cellphone_access = st.selectbox("Cellphone Access", cellphone_encoder.classes_)
    household_size = st.number_input("Household Size", 1, 20, step=1)
    age_of_respondent = st.number_input("Age of Respondent", 10, 100, step=1)
    gender = st.selectbox("Gender", gender_encoder.classes_)
    relationship = st.selectbox("Relationship with Head", relationship_encoder.classes_)
    marital_status = st.selectbox("Marital Status", marital_encoder.classes_)
    education = st.selectbox("Education Level", education_encoder.classes_)
    job_type = st.selectbox("Job Type", job_encoder.classes_)

#  Encode Input 
inputs = [
    country_encoder.transform([country])[0],
    year,
    location_encoder.transform([location_type])[0],
    cellphone_encoder.transform([cellphone_access])[0],
    household_size,
    age_of_respondent,
    gender_encoder.transform([gender])[0],
    relationship_encoder.transform([relationship])[0],
    marital_encoder.transform([marital_status])[0],
    education_encoder.transform([education])[0],
    job_encoder.transform([job_type])[0]
]

# Create Input DataFrame 
input_df = pd.DataFrame([inputs], columns=model_columns)

# Prediction
if st.sidebar.button("Predict Bank Account Ownership"):
    st.subheader("Input Summary")
    st.dataframe(input_df)

    prediction = model.predict(input_df)[0]
    st.subheader(" Prediction Result")
    if prediction == 1:
        st.success("This user is likely to **have a bank account**.")
    else:
        st.warning("This user is **unlikely to have a bank account**.")
