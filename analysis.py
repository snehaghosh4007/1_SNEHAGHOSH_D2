import pandas as pd

df = pd.read_csv("ifood_df.csv")

print(df.head())
# Check duplicate rows
print("Duplicate rows:")
print(df.duplicated().sum())


# Check missing values
print("Missing values:")
print(df.isnull().sum())
# Basic statistics
print("Basic Statistics:")
print(df.describe())
# Remove duplicate rows
df = df.drop_duplicates()

print("After removing duplicates:")
print(df.shape)
# Select important features for clustering

X = df[["Income", "MntTotal"]]

print(X.head())
from sklearn.cluster import KMeans

# Creating model
kmeans = KMeans(n_clusters=3, random_state=42)

# Making clusters
df["Cluster"] = kmeans.fit_predict(X)

print(df[["Income", "MntTotal", "Cluster"]].head())
import matplotlib.pyplot as plt

plt.scatter(
    df["Income"],
    df["MntTotal"],
    c=df["Cluster"]
)

plt.xlabel("Income")
plt.ylabel("Total Spending")

plt.title("Customer Segmentation")

plt.show()
print(df.groupby("Cluster")[["Income","MntTotal"]].mean())
print(df.groupby("Cluster")[["Income","MntTotal"]].mean())
print(df.groupby("Cluster")[["Income","MntTotal"]].mean())
print(df.groupby("Cluster")[["Income","MntTotal"]].mean())