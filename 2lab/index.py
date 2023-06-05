from nltk.tokenize import sent_tokenize

from nltk.tokenize import regexp_tokenize
from nltk.stem.snowball import EnglishStemmer

from nltk.corpus import brown

import nltk

#task 1
with open('text5.txt', 'r') as fp:
    data = fp.read()

    # a)
    sentences = sent_tokenize(data)
    print(len(sentences))

    # б)
    words = regexp_tokenize(data, pattern='\w+')

    common = nltk.FreqDist(words).most_common(9)

    print([word for word, count in common])

    # в)

    text = sentences[0]
    stemmer = EnglishStemmer()
    tokens=regexp_tokenize(text, pattern='\w+')
    stemmed=[]
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    result = " ".join(stemmed)

    print(result)

    #task 2

    # a)

    print(brown.fileids('mystery'))

    sents = [' '.join(i) for i in brown.sents(fileids='cl02')]
    print(sents[1:5])

    #б)
    adjectives = [i[0] for i in brown.tagged_words(fileids='cl02') if i[1] == 'JJ']

    common = nltk.FreqDist(adjectives).most_common(10)

    print([word for word, count in common])
