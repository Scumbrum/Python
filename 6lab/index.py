import pandas as pd
import re
import numpy as np
from nltk.stem.snowball import EnglishStemmer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from textblob import TextBlob
import random


if __name__ == '__main__':
    filename = "movie2.csv"

    dataset = pd.read_csv(filename)

    labels = np.array(dataset['label'])
    text = np.array(dataset['text'])

    filteredText = []
    stemmer = EnglishStemmer()
    from sklearn.feature_extraction.text import CountVectorizer

    for row in text:
        words = []
        for word in row.split(' '):
            regexDigit = re.compile(r'.*\d+.*')
            special = re.compile(r'.*[&*#$%~<>].*')
            if (not regexDigit.match(word)
                and not special.match(word)
            ):
                words.append(stemmer.stem(word))
        filteredText.append(' '.join(words))   

    wordBag = CountVectorizer(stop_words=stopwords.words('english'))

    train_corpus, test_corpus, train_label_names, test_label_names = train_test_split(filteredText, labels,test_size=0.3, random_state=0)

    cv_train_features = wordBag.fit_transform(train_corpus)
    cv_test_features = wordBag.transform(test_corpus)
    lr = LogisticRegression(penalty='l2', max_iter=100, C=1)

    lr.fit(cv_train_features, train_label_names)

    y_pred = lr.predict(cv_test_features)

    confusion = confusion_matrix(test_label_names, y_pred)
    accuracy = accuracy_score(test_label_names, y_pred)

    sentiments = []
    for review in test_corpus:
        blob = TextBlob(review)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.1:
            sentiments.append(1)
        else:
            sentiments.append(0)
    print(test_label_names)
    confusionTextBlob = confusion_matrix(test_label_names, sentiments)
    accuracyTextBlob = accuracy_score(test_label_names, sentiments)

    print("TextBlob:")
    print("Матриця невідповідностей:\n", confusionTextBlob)
    print("Точність моделі: {:.2f}%".format(accuracyTextBlob * 100))

    print("Логістина регресія:")
    print("Матриця невідповідностей:\n", confusion)
    print("Точність моделі: {:.2f}%".format(accuracy * 100))

    for i in range (3):
        bow = random.randrange(0, len(test_corpus))
        blob = TextBlob(test_corpus[bow])
        sentiment = blob.sentiment.polarity
        print("TextBlob: " + test_corpus[bow])
        if sentiment > 0.1:
            print("Semantic: Positive")
        else:
            print("Semantic: Negative")
        
        sentiment = lr.predict(cv_test_features[bow])
        print("Regression: " + test_corpus[bow])
        if sentiment[0] == 1:
            print("Semantic: Positive")
        else:
            print("Semantic: Negative")
            
        print()