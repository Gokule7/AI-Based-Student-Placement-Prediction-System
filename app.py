import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Placement Predictor", layout="centered")

st.title("🎓 AI-Based Student Placement Prediction System")
st.markdown("---")

if not os.path.exists('dataset/placement_model.pkl'):
    df = pd.read_csv('dataset/student_placement_data.csv')
    X = df.drop('Placed', axis=1)
    y = df['Placed']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open('dataset/placement_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('dataset/placement_model.pkl', 'rb') as f:
        model = pickle.load(f)

col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
    internships = st.number_input("Internships Count", 0, 10, 1)
    coding_skill = st.slider("Coding Skill (1-10)", 1, 10, 5)

with col2:
    communication = st.slider("Communication Score (1-10)", 1, 10, 5)
    projects = st.number_input("Projects Completed", 0, 20, 2)
    backlogs = st.number_input("Backlogs", 0, 10, 0)

st.markdown("---")

if st.button("Predict Placement", type="primary"):
    features = np.array([[cgpa, internships, coding_skill, communication, projects, backlogs]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    st.markdown("### Prediction Result")
    
    if prediction == 1:
        st.success("✅ **Prediction: PLACED**")
        confidence = probability[1] * 100
    else:
        st.error("❌ **Prediction: NOT PLACED**")
        confidence = probability[0] * 100
    
    st.metric("Confidence Score", f"{confidence:.2f}%")
    
    st.markdown("---")
    st.markdown("### Probability Distribution")
    col_a, col_b = st.columns(2)
    col_a.metric("Not Placed", f"{probability[0] * 100:.2f}%")
    col_b.metric("Placed", f"{probability[1] * 100:.2f}%")
