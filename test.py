import pandas as pd

from src.preprocess import DataPreprocessor

print("Loading Dataset...")

data = pd.read_csv(
    "dataset/linkedin_dataset_10000_smart.csv"
)

processor = DataPreprocessor()

X, y = processor.preprocess(data)

print()

print("Dataset Loaded Successfully")

print()

print("Feature Matrix Shape :", X.shape)

print("Target Shape :", y.shape)

print()

print(y.value_counts())