import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load dataset
df = pd.read_csv("tweets.csv")

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Function to classify sentiment
def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df['Sentiment'] = df['tweet'].apply(get_sentiment)

# Show results
print("\n📊 Sentiment Analysis Results:\n")
print(df)

# Save output to CSV
df.to_csv("tweets_with_sentiment.csv", index=False)

# Count values
counts = df['Sentiment'].value_counts()

print("\n📌 Summary:")
print(counts)

# Visualization
plt.figure(figsize=(6,4))
counts.plot(kind='bar')
plt.title("Tweet Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()