"""
LDA and NMP analysis Adapted from:
    http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
"""

from __future__ import print_function

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def main():
    n_features = 1000
    n_topics = 5
    n_top_words = 10
    
    df = pd.read_csv('../data/Lyrics_enriched.csv')
    df['lyric_length'] = df.Lyrics.map(lambda lyric: len(lyric.split()))
    data_samples = df.Lyrics.values.tolist()
    
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(data_samples)
    nmf = NMF(n_components=n_topics, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print("NMF:")
    print_top_words(nmf, tfidf_feature_names, n_top_words)
    
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print("LDA")
    print_top_words(lda, tf_feature_names, n_top_words)
    
    #Added TF and TF-IDF output
    print("Top TF-IDF words:")
    zipped = zip(tfidf.data, tfidf_feature_names)
    print(" ".join([x[1] for x in sorted(zipped, reverse=True, key=(lambda x: x[0]))[0:10]]))
    print("")
    
    print("Top TF words:")
    zipped = zip(tf.data, tf_vectorizer.get_feature_names())
    print(" ".join([x[1] for x in sorted(zipped, reverse=True, key=(lambda x: x[0]))[0:10]]))
    
if __name__ == "__main__":
    main()