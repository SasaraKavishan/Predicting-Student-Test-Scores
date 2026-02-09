import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

with open("style.css", "r") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

model_xgb = joblib.load('xgboost_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

st.title('Exam Score Predictor')
st.markdown('Enter student details')

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', 17, 24, 20)
    gender = st.selectbox('Gender', ['male', 'female', 'other'])
    course = st.selectbox('Course', ['ba', 'bca', 'b.com', 'b.sc', 'b.tech', 'bba', 'diploma'])
    study_hours = st.number_input('Study Hours (per day)', 0.0, 10.0, 5.0, step=0.1)
    class_attendance = st.number_input('Class Attendance (%)', 0.0, 100.0, 80.0, step=0.1)

with col2:
    internet_access = st.selectbox('Internet Access', ['yes', 'no'])
    sleep_hours = st.number_input('Sleep Hours (per night)', 0.0, 10.0, 7.0, step=0.1)
    sleep_quality = st.selectbox('Sleep Quality', ['poor', 'average', 'good'])
    study_method = st.selectbox('Study Method', ['group study', 'coaching', 'online videos', 'self-study', 'mixed'])
    facility_rating = st.selectbox('Facility Rating', ['low', 'medium', 'high'])
    exam_difficulty = st.selectbox('Exam Difficulty', ['easy', 'moderate', 'hard'])

if st.button('Predict Exam Score'):
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'course': [course],
        'study_hours': [study_hours],
        'class_attendance': [class_attendance],
        'internet_access': [internet_access],
        'sleep_hours': [sleep_hours],
        'sleep_quality': [sleep_quality],
        'study_method': [study_method],
        'facility_rating': [facility_rating],
        'exam_difficulty': [exam_difficulty]
    })

    input_preprocessed = preprocessor.transform(input_data)
    prediction = model_xgb.predict(input_preprocessed)[0]

    st.success(f'**Predicted Exam Score: {prediction:.2f}**')


