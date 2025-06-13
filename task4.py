import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("twitter_training.csv", header=None)
df.columns = ["entity", "sentiment", "blank", "text"]

# Display basic info
print("Dataset preview:\n", df.head())

# Drop unnecessary column
df.drop("blank", axis=1, inplace=True)

# Check sentiment distribution
print("\nSentiment value counts:\n", df['sentiment'].value_counts())

# Visualization: Sentiment Distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="sentiment", order=df['sentiment'].value_counts().index, palette="Set2")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.show()

# Visualization: Sentiment by Entity (Top 5 entities)
top_entities = df['entity'].value_counts().head(5).index
df_top = df[df['entity'].isin(top_entities)]

plt.figure(figsize=(10,6))
sns.countplot(data=df_top, x="entity", hue="sentiment", palette="Set1")
plt.title("Sentiment Breakdown by Top 5 Entities")
plt.xlabel("Entity")
plt.ylabel("Tweet Count")
plt.legend(title="Sentiment")
plt.show()
