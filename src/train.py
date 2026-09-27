import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from preprocess import DataPreprocessor


print("Loading Dataset...\n")

data = pd.read_csv(
    "dataset/linkedin_dataset_10000_smart.csv"
)

processor = DataPreprocessor()

X, y = processor.preprocess(data)
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("Preprocessing Complete\n")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("Training Random Forest...\n")

model = XGBClassifier(

    n_estimators=300,

    max_depth=6,

    learning_rate=0.1,

    random_state=42,

    eval_metric="mlogloss"

)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

precision = precision_score(

    y_test,

    prediction,

    average="macro"

)

recall = recall_score(

    y_test,

    prediction,

    average="macro"

)

f1 = f1_score(

    y_test,

    prediction,

    average="macro"

)

print("="*60)

print("MODEL PERFORMANCE")

print("="*60)

print(f"Accuracy  : {accuracy*100:.2f}%")

print(f"Precision : {precision*100:.2f}%")

print(f"Recall    : {recall*100:.2f}%")

print(f"F1 Score  : {f1*100:.2f}%")

print()

prediction = label_encoder.inverse_transform(prediction)

y_test = label_encoder.inverse_transform(y_test)

print(classification_report(

    y_test,

    prediction

))

cm = confusion_matrix(

    y_test,

    prediction

)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,
    display_labels=label_encoder.classes_

)

disp.plot(cmap="Blues")

plt.title("Random Forest Confusion Matrix")

os.makedirs("outputs", exist_ok=True)

plt.savefig("outputs/confusion_matrix.png")

plt.show()

os.makedirs("models", exist_ok=True)


joblib.dump(model, "models/xgboost.pkl")

joblib.dump(processor, "models/preprocessor.pkl")

print("\nModel Saved Successfully.")