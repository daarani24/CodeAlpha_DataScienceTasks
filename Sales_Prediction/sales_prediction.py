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

X=df[["Tv","Radio","Newspaper"]]
y=df["Sales"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("\n Training Samples:",len(X_train))
print("Testing Sample:",len(X_test))

model=LinearRegression()
model.fit(X_train,X_test)

y_pred=model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"Accuracy: {r2*100:.2f}%")

plt.figure(figsize=(7,5))
plt.scatter(y_test,y_pred)
plt.plot([y_test.min(),y_test.max()], [y_test.min(),y_test.max()],'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

imp=pd.DataFrame({"Channel":X.columns,"Coefficient": model.coef_})
print("\nAdvertising Impact:")
print(imp)
plt.figure(figsize=(6,4))
sns.barplot(data=imp,x="Channel",y="Coefficient")
plt.title("Advertising Channel Impact")
plt.show()

sample=X_test.iloc[[0]]
predicted_sales=model.predict(sample)
print("Sample Predcition")
print("Predicted Sales:",predicted_sales[0])
print("Actual Sales:",y_test.iloc[0])