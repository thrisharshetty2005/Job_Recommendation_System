# Job Recommendation System

## Overview

The Job Recommendation System is a machine learning-based application that helps users find suitable job opportunities based on their skills and preferences. The system analyzes job-related information and identifies jobs that closely match the skills provided by the user.

It also includes a salary category prediction module that uses machine learning to classify salary levels.

## Key Features

- Job data preprocessing and cleaning
- Skill-based job recommendation
- TF-IDF for converting job skills into numerical features
- Cosine Similarity for finding similar jobs
- Salary category prediction using Random Forest
- Confusion matrix for model evaluation
- Feature importance analysis
- Comparison of machine learning models
- Accuracy visualization

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## How the System Works

1. Job data is collected and preprocessed.
2. Relevant skill information is extracted from the dataset.
3. TF-IDF is used to represent job skills as numerical vectors.
4. Cosine Similarity compares the user's skills with available job listings.
5. Jobs with higher similarity are recommended to the user.
6. A Random Forest model is used to predict the salary category.
7. The models are evaluated using different performance metrics.

## Dataset

The project uses a synthetic LinkedIn-style job dataset containing:

- 10,000 job records
- Balanced salary categories
- Job titles
- Companies
- Skills
- Locations
- Salary-related information

## Model Performance

The implemented machine learning model achieved the following results:

| Metric | Score |
|---|---:|
| Accuracy | 97.70% |
| Precision | 92.09% |
| Recall | 95.61% |
| F1 Score | 93.74% |

## Project Structure

```text
Job_Recommendation_System/
│
├── dataset/          # Dataset files
├── models/           # Trained machine learning models
├── outputs/          # Generated results and graphs
├── src/              # Source code
├── app.py            # Main application
├── recommend_test.py # Recommendation testing
├── test.py           # Model testing
├── requirements.txt  # Required Python packages
└── README.md         # Project documentation
