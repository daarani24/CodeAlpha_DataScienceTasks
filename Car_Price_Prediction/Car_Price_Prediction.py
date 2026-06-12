import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df=pd.read_csv("Car_Price_Prediction/car data.csv")
print("shape:",df.shape)
print(df.head())
print(df.info())

print("Missing values:")
print(df.isnull().sum())

df.dropna(inplace=True)

cur_year=2025

df["Car_Age"]=cur_year-df["Year"]

if "Year" in df.columns:
    df.dropna("year",axis=1,inplace=True)

if "Car_Name" in df.columns:
    df.drop("Car_Name",axis=1,inplace=True)

le=LabelEncoder()
for c in df.select_dtypes(include="object").columns:
    df[c]=le.fit_transform(df[c])

print("Processed Dataset:")
print(df.head())

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Selling_Price"],bins=25,kde=True)
plt.title("Selling Price Distribution")
plt.xlabel("Selling Price")
plt.show()

X=df.drop("Selling_Price",axis=1)
y=df["Selling_Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

print("Training Samples:",len(X_train))
print("Testing Samples:",len(X_test))

model=RandomForestRegressor(n_estimators=200,random_state=42)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print(f"Mean Absolute Error:{mae:.2f}")
print(f"Mean Squared Error:{mse:.2f}")
print(f"Root mean Squared Error:{rmse:.2f}")
print(f"R2 Score:{r2:.4f}")

plt.figure(figsize=(8,5))
plt.scatter(y_test,y_pred,color="red",alpha=0.7)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'b--')

plt.xlabel("actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted car Price")
plt.show()

imp=pd.DataFrame({"Feature":X.columns,"Importance":model.feature_importances_})
imp=imp.sort_values(by="Importance",ascending=False)
print("Feature Importance:")
print(imp)
plt.figure(figsize=(8,5))
sns.barplot(data=imp,x="Importance",y="Feature")
plt.title("Feature Importance")
plt.show()

s=X_test.iloc[[0]]
pred_price=model.predict(s)
print("Predicted car price:",pred_price[0])
print("Actual car price:",y_test.iloc[0])

