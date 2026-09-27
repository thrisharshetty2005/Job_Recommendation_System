import joblib
import matplotlib.pyplot as plt
import pandas as pd

model = joblib.load("models/random_forest.pkl")

importances = model.feature_importances_

feature_names = []

feature_names.extend([
    "Age",
    "Experience",
    "Job Title",
    "Company",
    "Location",
    "Education"
])

remaining = len(importances) - len(feature_names)

for i in range(remaining):
    feature_names.append(f"Skill_{i+1}")

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"][:15],
    importance["Importance"][:15]
)

plt.title("Top 15 Important Features")

plt.xlabel("Importance")

plt.tight_layout()

plt.savefig("outputs/feature_importance.png")

plt.show()