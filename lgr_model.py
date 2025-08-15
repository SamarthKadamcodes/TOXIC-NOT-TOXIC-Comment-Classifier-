import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("jigsaw_multilingual_toxicity.csv")
df = df.dropna(subset=['comment_text', 'toxic'])

X = df['comment_text']
y = df['toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

y_predict = model.predict(X_test_vec)

print(classification_report(y_test, y_predict))

sample_comment = ["I hate you, you are awful!"]
sample_vec = vectorizer.transform(sample_comment)
prediction = model.predict(sample_vec)
print("Toxic" if prediction[0] == 1 else "Not Toxic")