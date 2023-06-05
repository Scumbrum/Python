from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from gensim.models import word2vec
import nltk

with open('doc15.txt', 'r') as fp:
    data = fp.read()

    # 1)

    corpus = data.split('\n')
    
    bigramAndUnigram = CountVectorizer(ngram_range=(1,2))

    matrix = bigramAndUnigram.fit_transform(corpus)

    word_index = bigramAndUnigram.vocabulary_['laser']

    laser_vector = matrix[:, word_index].toarray().flatten()

    print(laser_vector)

    # 2)        

    vectorizer = TfidfVectorizer()
    tdIdf = vectorizer.fit_transform(corpus)

    n_clusters = 2
    linkage = 'ward'
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    cluster_model.fit(tdIdf.toarray())

    for i in range(len(corpus)):
        print("Document {}: cluster {}".format(i, cluster_model.labels_[i]))

    # 3)  

    wpt = nltk.regexp_tokenize
    tokenized_corpus = [wpt(document, pattern='\w+') for document in corpus]
    feature_size = 100
    window_context = 30
    min_word_count = 1
    sample = 1e-3

    w2v_model = word2vec.Word2Vec(tokenized_corpus,
    vector_size=feature_size,window=window_context,
    min_count=min_word_count,sample=sample)
    print(w2v_model.wv.most_similar('chocolate'))
    print(w2v_model.wv.most_similar('link'))