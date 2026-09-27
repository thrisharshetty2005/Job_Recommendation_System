from src.recommend import JobRecommender

recommender = JobRecommender(
    "dataset/linkedin_dataset_10000_smart.csv"
)

print("=" * 60)
print("JOB RECOMMENDATION SYSTEM")
print("=" * 60)

skills = input("\nEnter your skills:\n\n")

results = recommender.recommend_jobs(
    skills,
    top_n=5
)

print("\nTop 5 Recommended Jobs\n")

print(results.to_string(index=False))