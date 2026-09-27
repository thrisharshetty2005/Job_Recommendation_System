import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


class DataPreprocessor:

    def __init__(self):

        self.job_encoder = LabelEncoder()
        self.company_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.education_encoder = LabelEncoder()

        self.scaler = StandardScaler()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=100
        )

    def preprocess(self, dataframe):

        dataframe = dataframe.copy()

        dataframe.fillna("", inplace=True)

        dataframe["job_title_enc"] = self.job_encoder.fit_transform(
            dataframe["job_title"]
        )

        dataframe["company_enc"] = self.company_encoder.fit_transform(
            dataframe["company"]
        )

        dataframe["location_enc"] = self.location_encoder.fit_transform(
            dataframe["location"]
        )

        dataframe["education_enc"] = self.education_encoder.fit_transform(
            dataframe["education"]
        )

        skills = self.vectorizer.fit_transform(
            dataframe["job_skills"]
        ).toarray()

        numeric = dataframe[
            [
                "experience_years",
                "age",
                "job_title_enc",
                "company_enc",
                "location_enc",
                "education_enc"
            ]
        ].values

        X = np.hstack((numeric, skills))

        X = self.scaler.fit_transform(X)

        y = dataframe["salary_category"]

        return X, y