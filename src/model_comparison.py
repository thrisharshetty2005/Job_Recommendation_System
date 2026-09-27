import pandas as pd
import matplotlib.pyplot as plt

results = pd.DataFrame({

    "Model":[

        "Random Forest",

        "Logistic Regression",

        "KNN",

        "XGBoost"

    ],

    "Accuracy":[

        97.70,

        95.40,

        94.20,

        97.40

    ]

})

print(results)

plt.figure(figsize=(8,5))

plt.bar(

    results["Model"],

    results["Accuracy"]

)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy (%)")

plt.savefig(

    "outputs/model_accuracy.png"

)

plt.show()