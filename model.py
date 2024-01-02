#Final project on phishing website detection
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
#Read data from dataset
data=pd.read_csv("dataset_phishing.csv")
#Encoding the string columns
#Encoding the 'url' column
encoder = LabelEncoder()
data['url'] = encoder.fit_transform(data['url'])
#Encoding the status column 
le=LabelEncoder()
data["status"]=le.fit_transform(data["status"])
# Selecting features and target variable
# Excluding status columns
X = data.drop(['status'], axis=1) 
y = data['status']
# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Training the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Making predictions
predictions = model.predict(X_test)
# Evaluating the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")