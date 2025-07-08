import pandas as pd
from transformers import pipeline

# Load sentiment-analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Load legal feedback from CSV
df = pd.read_csv("legal_feedback.csv")

# Analyze sentiment
df["Sentiment"] = df["Text"].apply(lambda x: sentiment_pipeline(x)[0]["label"])

# Save output
df.to_csv("legal_feedback_labeled.csv", index=False)
print("Sentiment analysis complete. Output saved to 'legal_feedback_labeled.csv'")
