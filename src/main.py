import joblib
import pandas as pd

from recommend import JobRecommender

print("=" * 60)
print("        JOB RECOMMENDATION SYSTEM")
print("=" * 60)

# Load saved model
model = joblib.load("models/random_forest.pkl")

# Load recommender
recommender = JobRecommender(
    "dataset/linkedin_dataset_10000_smart.csv"
)

while True:

    print("\n")
    print("=" * 60)
    print("1. Recommend Jobs")
    print("2. Exit")
    print("=" * 60)

    choice = input("Enter your choice: ")

    if choice == "1":

        skills = input("\nEnter your skills:\n")

        recommendations = recommender.recommend_jobs(
            skills,
            top_n=5
        )

        print("\nTop 5 Recommended Jobs\n")

        print(recommendations.to_string(index=False))

    elif choice == "2":

        print("\nThank you for using Job Recommendation System.")
        break

    else:

        print("\nInvalid Choice.")