"""
Streamlit app - Wellness Tourism Package purchase predictor.

On startup, the app downloads the trained model from the Hugging Face Model
Hub (falling back to a local copy if present, useful for local testing) and
serves an interactive form so a salesperson can enter a customer's details
and get a purchase-likelihood prediction before making contact.
"""

import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "your-hf-username/tourism-wellness-package-model")
LOCAL_FALLBACK_PATH = "best_model.joblib"


@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="best_model.joblib")
    except Exception:
        model_path = LOCAL_FALLBACK_PATH
    return joblib.load(model_path)


st.set_page_config(page_title="Wellness Package Predictor", page_icon="\U0001f9d8", layout="centered")
st.title("\U0001f9d8 Wellness Tourism Package - Purchase Predictor")
st.write(
    "Enter a prospective customer's details to estimate the likelihood "
    "they will purchase the new Wellness Tourism Package."
)

model = load_model()

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)
        occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])

    with col2:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        num_trips = st.number_input("Avg. Number of Trips per Year", min_value=0, max_value=20, value=3)
        passport = st.selectbox("Holds Passport?", ["Yes", "No"])
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
        num_children = st.number_input("Number of Children Visiting (<5 yrs)", min_value=0, max_value=5, value=0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_person_visiting,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    proba = model.predict_proba(input_df)[0, 1]
    prediction = model.predict(input_df)[0]

    st.subheader("Result")
    if prediction == 1:
        st.success(f"\u2705 Likely to purchase \u2014 estimated probability: {proba:.1%}")
    else:
        st.warning(f"\u274c Unlikely to purchase \u2014 estimated probability: {proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))
