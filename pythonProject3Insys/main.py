# Import necessary libraries

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('social_media_data.csv')
df['text'] = df['text'].str.lower().str.replace('[^a-z]', ' ')
X = df['text']
y = df['sentiment']

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

def predict_sentiment(text):
    text = text.lower()
    text = vectorizer.transform([text])
    return clf.predict(text)[0]