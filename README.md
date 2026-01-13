# 🔹 AI-Based Student Placement Prediction System

## 📌 Problem Statement
Many students don't know whether they are placement-ready. This system predicts placement chances using academic + skill data.

## 📊 Inputs (Features)
- CGPA
- Internships count
- Coding skill level
- Communication score
- Projects completed
- Backlogs

## 🎯 Output
- Placed or Not Placed
- Probability score (confidence)

## 🚀 Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the application:
```bash
streamlit run app.py
```

## 📁 Project Structure
```
├── dataset/
│   ├── student_placement_data.csv
│   └── placement_model.pkl
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```
