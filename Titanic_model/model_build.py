#----IMPORT LIBRARIES----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#----TO SAVE OUR MODEL----
import pickle

#----LOAD DATA----
df = pd.read_csv(r"C:\Users\Dell\OneDrive\New folder\AIML(all basics)\Titanic_model\Titanic.csv")
print("First 5 rows of dataset:")
print(df.head())

#----DROP UNNECESSARY COLUMNS----
df = df.drop(columns=[col for col in ['PassengerId', 'Name', 'Ticket', 'Cabin'] if col in df.columns])

#----HANDLE MISSING VALUES----
# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with most frequent value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Print to check if anything is still missing
print("\nMissing values before full check:\n", df.isnull().sum())

# Fill any remaining missing numerical values just in case
df.fillna(df.median(numeric_only=True), inplace=True)

#----ENCODE CATEGORICAL VARIABLES----
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

#----CHECK AGAIN FOR MISSING DATA----
print("\nMissing values after filling:\n", df.isnull().sum())

#----TARGET AND FEATURES----
X = df.drop('Survived', axis=1)
y = df['Survived']

#----SPLIT DATA INTO TRAIN & TEST----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#----BUILD AND TRAIN MODEL----
model = LogisticRegression()
model.fit(X_train, y_train)

#----MAKE PREDICTIONS----
y_pred = model.predict(X_test)

#----EVALUATE MODEL----
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#----VISUALIZE FEATURE IMPORTANCE----
features = X.columns
importance = model.coef_[0]

plt.figure(figsize=(10,6))
sns.barplot(x=features, y=importance)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#----SAVE MODEL----
with open('titanic_model.pkl', 'wb') as file:
    pickle.dump(model, file)

#----SAVE FEATURES COLS USED DURING TRAINING----
with open('model_features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)


