# Spam Email Detection Using Logistic Regression

## Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df.isnull().sum()
# Drop unnecessary columns
df = df[['v1', 'v2']]  # 'v1' -> Label (spam/ham), 'v2' -> Message

# Rename the columns for clarity
df.columns = ['label', 'message']

# Check for missing values
df.isnull().sum()
# Initialize CountVectorizer to convert text into features
vectorizer = CountVectorizer(stop_words='english')

# Transform the messages into a sparse matrix of token counts
X = vectorizer.fit_transform(df['message'])

# Target variable (labels)
y = df['label'].map({'ham': 0, 'spam': 1})
# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the resulting datasets
print(X_train.shape, X_test.shape)
# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model using the training data
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Display classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
# Predict on a new message
new_messages = ['Free lottery tickets waiting for you', 'Hey, are we still meeting tomorrow?']

# Transform the messages to the same format as the training data
new_data = vectorizer.transform(new_messages)

# Make predictions
predictions = model.predict(new_data)

# Display the predictions
for message, label in zip(new_messages, predictions):
    print(f"Message: {message} - Predicted as: {'Spam' if label == 1 else 'Ham'}")
