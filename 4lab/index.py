import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.snowball import EnglishStemmer

filename = "news.csv"

dataset = pd.read_csv(filename)
text = np.array(dataset['text'])
filteredText = []
stemmer = EnglishStemmer()

for row in text:
    words = []
    for word in row.split(' '):
        regexLink = re.compile(r'.*http.*')
        regexDigit = re.compile(r'.*\d+.*')
        special = re.compile(r'.*[&*#$%~].*')
        if (not regexLink.match(word)
            and not regexDigit.match(word)
            and not special.match(word)
        ):
            words.append(stemmer.stem(word))
    filteredText.append(' '.join(words))   

wordBag = CountVectorizer(stop_words=stopwords.words('english'))

train_corpus, test_corpus, train_label_names, test_label_names = train_test_split(filteredText, np.array(dataset['label']),test_size=0.3, random_state=0)

cv_train_features = wordBag.fit_transform(train_corpus)
cv_test_features = wordBag.transform(test_corpus)

svm = LinearSVC(penalty='l2', C=1, random_state=0)
rfc = RandomForestClassifier(n_estimators=10, random_state=0)

svm.fit(cv_train_features, train_label_names)
rfc.fit(cv_train_features, train_label_names)

predictedLinear = svm.predict(cv_test_features)
predictedRandom = rfc.predict(cv_test_features)
accuracyRandom = accuracy_score(test_label_names, predictedRandom)
accuracyLinear = accuracy_score(test_label_names, predictedLinear)
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l2', 'l1'], 'random_state': [0, 42]}
betterSvm = GridSearchCV(LinearSVC(), parameters)

betterSvm.fit(cv_train_features, train_label_names)

predictedBetterLinear = betterSvm.predict(cv_test_features)
accuracyBetterLinear = accuracy_score(test_label_names, predictedBetterLinear)

print("Accuracy linear:", accuracyLinear)
print("Accuracy random:", accuracyRandom)
print("Accuracy better linear:", accuracyBetterLinear)
