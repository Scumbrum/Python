import pandas as pd

from nltk.corpus import stopwords
import re
import gensim
import numpy as np
import random


if __name__ == '__main__':

    filename = "news.csv"

    dataset = pd.read_csv(filename)

    labels = np.array(dataset['label'])
    text = np.array(dataset['text'])
    filteredText = []


    for row in text:
        words = []
        for word in row.split(' '):
            regexLink = re.compile(r'.*http.*')
            regexDigit = re.compile(r'.*\d+.*')
            special = re.compile(r'.*[\-&*#$%~].*')
            stop_words = set(stopwords.words('english'))
            if (not regexLink.match(word)
                and not regexDigit.match(word)
                and not special.match(word)
                and word.lower() not in stop_words
                and len(word) > 0
            ):
                words.append(word.lower())
        filteredText.append(' '.join(words))     



    filteredText = [row.strip().split(' ') for row in filteredText]
    bigram = gensim.models.Phrases(filteredText, min_count=20,
    threshold=20, delimiter='_')
    bigram_model = gensim.models.phrases.Phraser(bigram)
    norm_corpus_bigrams = [bigram_model[doc] for doc in filteredText]
    dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
    dictionary.filter_extremes(no_below=20, no_above=0.6)

    corpus = [dictionary.doc2bow(doc) for doc in norm_corpus_bigrams]
    avg_coherence_cvs  = []
    avg_coherence_umasses = []
    perplaxities = []

    num_topics = [2, 3]
    for n in num_topics:
        lda_model = gensim.models.LdaModel(corpus=corpus,
        id2word=dictionary, chunksize=1740, alpha='auto', eta='auto',
        random_state=0, iterations=500, num_topics=n,
        passes=20, eval_every=None)
        cv_coherence_model_lda = gensim.models.CoherenceModel(
        model=lda_model, corpus=corpus, coherence='c_v', texts=norm_corpus_bigrams, dictionary=dictionary)
        avg_coherence_cv = cv_coherence_model_lda.get_coherence()
        umass_coherence_model_lda = gensim.models.CoherenceModel(
        model=lda_model, corpus=corpus, coherence='u_mass', texts=norm_corpus_bigrams, dictionary=dictionary)
        avg_coherence_umass = umass_coherence_model_lda.get_coherence()
        perplexity = lda_model.log_perplexity(corpus)   
        print("Number of topics =", n)
        print("CV  =", avg_coherence_cv)
        print("UMASS =", avg_coherence_umass)
        print("Perplaxity",  perplexity)
        avg_coherence_cvs.append(avg_coherence_cv)
        avg_coherence_umasses.append(avg_coherence_umass)
        perplaxities.append(perplexity)
        print()

    bestScore = float('-inf')
    bestTopicNumber = 1
    for index in range(len(num_topics)):
        score = avg_coherence_cvs[index] / (abs(perplaxities[index]) + abs(avg_coherence_umasses[index]))
        if score > bestScore:
            bestScore = score
            bestTopicNumber = num_topics[index]

    print("Best number of topics: ", bestTopicNumber)

    lda_model = gensim.models.LdaModel(corpus=corpus,
        id2word=dictionary, chunksize=1740, alpha='auto', eta='auto',
        random_state=0, iterations=500, num_topics=bestTopicNumber,
        passes=20, eval_every=None)
    for i in range(4):
        bow = random.randrange(0, len(corpus))
        topics = lda_model.get_document_topics(corpus[bow])
        print("Document: ", bow)
        topics_words = [(tp, [dictionary[wd[0]] for wd in lda_model.get_topic_terms(tp[0])]) for tp in topics]
        for topic,words in topics_words:
            print(str(topic[0])+ "::"+str(topic[1]) + "::" + str(words))
        print()

    import nltk
    from nltk.corpus import gutenberg


    persuasion = gutenberg.words('austen-persuasion.txt')

    filteredText = []

    for word in persuasion:
            
        regexDigit = re.compile(r'.*\d+.*')
        special = re.compile(r'.*[,;\-\'"!?&*#$%~.].*')
        stop_words = set(stopwords.words('english'))
        if (not regexDigit.match(word)
            and not special.match(word)
            and word not in stop_words
        ):
            filteredText.append(word) 

    trigrams = list(nltk.trigrams(filteredText))


    freq_dist = nltk.FreqDist(trigrams)


    for trigram, frequency in freq_dist.most_common(5):
        print(trigram, frequency)
