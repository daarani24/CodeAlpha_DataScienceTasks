import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv("Sales_Prediction/Advertising.csv")
print("Dataset Shape:",df.shape)
print("\nFirst 5 record:")
print(df.head())

print("\nDataset Information:")
print(df.info())
print("\n Missing Values:")
print(df.isnull().sum())

plt.figure(figsize=(8,5))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["TV"],df["Sales"])
plt.xlabel("Tv Advertisement Spend")
plt.ylabel("Sales")
plt.title("TV Spend vs Sales")
plt.show()


