import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class JobRecommender:

    def __init__(self, dataset_path):

        self.data = pd.read_csv(dataset_path)

        self.vectorizer = TfidfVectorizer(stop_words="english")

        self.skill_matrix = self.vectorizer.fit_transform(
            self.data["job_skills"]
        )

    def recommend_jobs(self, user_skills, top_n=5):

        user_vector = self.vectorizer.transform([user_skills])

        similarity = cosine_similarity(
            user_vector,
            self.skill_matrix
        )

        scores = similarity.flatten()

        top_indices = scores.argsort()[-top_n:][::-1]

        recommendations = self.data.iloc[
            top_indices
        ][
            [
                "job_title",
                "company",
                "location",
                "education",
                "experience_years",
                "salary",
                "salary_category",
                "job_skills"
            ]
        ]

        return recommendations