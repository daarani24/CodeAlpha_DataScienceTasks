import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

df=pd.read_csv("Iris_Flower_Classification/Iris.csv")

print("Shape:",df.shape)
print(df.head())

le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])

X=df.drop('Species',axis=1)
y=df['Species']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"\n Accuracy: {accuracy*100:.2f}%")

print("\n Classification Report:")
print(classification_report(y_test,y_pred,target_names=le.classes_))

cm=confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=le.classes_,yticklabels=le.classes_)

plt.xlabel("predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()