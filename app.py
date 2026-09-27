import streamlit as st
import pandas as pd
import joblib
import os

from src.recommend import JobRecommender
from src.resume_parser import extract_text, extract_skills

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="💼",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("💼 AI Job Recommendation System")
st.write("Find jobs based on your skills using Machine Learning.")

# -----------------------------
# Load Dataset
# -----------------------------
DATASET_PATH = "dataset/linkedin_dataset_10000_smart.csv"

if not os.path.exists(DATASET_PATH):
    st.error("Dataset not found!")
    st.stop()

data = pd.read_csv(DATASET_PATH)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "models/random_forest.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# -----------------------------
# Load Recommender
# -----------------------------
recommender = JobRecommender(DATASET_PATH)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "Home",

        "Recommend Jobs",

        "Resume Analyzer",

        "About"

    ]

)

# -----------------------------
# HOME
# -----------------------------
if page == "Home":
    
    st.header("🏠 Welcome to AI Job Recommendation System")

    st.markdown("""
This application uses **Machine Learning**, **TF-IDF**, and **Cosine Similarity**
to recommend jobs based on your skills.

### Features

✔ Job Recommendation

✔ Salary Category Prediction

✔ Random Forest Classification

✔ TF-IDF & Cosine Similarity

✔ Interactive Dashboard
""")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Size", "10,000")

    with col2:
        st.metric("Accuracy", "97.70%")

    with col3:
        st.metric("Models", "Random Forest")

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(data.head(10))

# -----------------------------
# JOB RECOMMENDATION
# -----------------------------
elif page == "Recommend Jobs":
    
    st.header("💼 Find Your Dream Job")

    skills = st.text_area(
        "Enter Your Skills",
        placeholder="Example: Python SQL Machine Learning"
    )

    top_jobs = st.slider(
        "Number of Recommendations",
        1,
        10,
        5
    )

    if st.button("🔍 Recommend Jobs"):

        if skills.strip() == "":

            st.warning("Please enter your skills.")

        else:

            recommendations = recommender.recommend_jobs(
                skills,
                top_n=top_jobs
            )

            st.success("Top Matching Jobs")

            for index, row in recommendations.iterrows():

                with st.container():

                    st.subheader(row["job_title"])

                    st.write(f"🏢 Company : {row['company']}")

                    st.write(f"📍 Location : {row['location']}")

                    st.write(f"🎓 Education : {row['education']}")

                    st.write(f"💼 Experience : {row['experience_years']} Years")

                    st.write(f"💰 Salary : ₹{row['salary']}")

                    st.write(f"📊 Category : {row['salary_category']}")

                    st.write(f"🛠 Skills : {row['job_skills']}")

                    st.divider()
                    
                    
elif page == "Resume Analyzer":
    
    st.header("📄 Resume Analyzer")

    uploaded_file = st.file_uploader(

        "Upload Resume (PDF)",

        type=["pdf"]

    )

    if uploaded_file is not None:

        text = extract_text(uploaded_file)

        skills = extract_skills(text)

        st.subheader("Detected Skills")

        st.write(skills)

        if len(skills) > 0:

            recommendations = recommender.recommend_jobs(

                " ".join(skills),

                top_n=5

            )

            st.subheader("Recommended Jobs")

            st.dataframe(recommendations)

        else:

            st.warning("No matching skills detected.")

# -----------------------------
# ABOUT
# -----------------------------
else:
    
    st.header("ℹ About Project")

    st.markdown("""

## AI Job Recommendation System

This project recommends jobs using Machine Learning.

### Algorithms Used

- Random Forest
- TF-IDF
- Cosine Similarity

### Technologies

- Python
- Streamlit
- Pandas
- Scikit-Learn
- Matplotlib

### Dataset

10,000 Job Records

Balanced Salary Categories

### Performance

Accuracy : **97.70%**

Precision : **92.09%**

Recall : **95.61%**

F1 Score : **93.74%**

""")