import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data from a CSV file
# Assume the CSV file has a "Text" column and a "Label" column for supervised learning
data = pd.read_csv("yourfilepath")  # Replace with your CSV file path

# Split data into features and labels
X = data["Text"]
y = data["Label"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predict sentiment labels for the test set
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy and display evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", confusion)

# Apply the model to the entire dataset and save the predictions to a new CSV file
data["Predicted_Sentiment"] = model.predict(vectorizer.transform(X))
data.to_csv("sentiment_predictions.csv", index=False)
print("\nPredictions saved to 'sentiment_predictions.csv'.")
