import pandas as pd

# Read the raw dataset (tab-separated)
df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "text"])

# Save as CSV for future use
df.to_csv("data/spam.csv", index=False)

print("Dataset saved as spam.csv")
print(df.head())
