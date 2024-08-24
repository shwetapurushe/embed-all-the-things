# from gensim.test.utils import common_texts
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_tags, strip_numeric, strip_punctuation, strip_non_alphanum, strip_multiple_whitespaces, strip_short, stem_text
import gensim.downloader as api
from gensim.models import Word2Vec

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

 # getting sentence embeddings from word embeddings
# simple averaging
# def sent_vec(sent):
#     vector_size = word2vec_pretrained.vector_size
#     wv_res = np.zeros(vector_size)
#     # print(wv_res)
#     ctr = 1
#     for w in sent:
#         if w in word2vec_pretrained:
#             ctr += 1
#             wv_res += word2vec_pretrained[w]
#     wv_res = wv_res/ctr
#     return wv_res

def load_data(dataset):
    # dataPath = os.path.join(os.getcwd(), dataset)
    # data = pd.read_csv(dataPath)
    data = pd.read_csv("/Users/shweta/Documents/embedding-all-the-things/imdb_dataset.csv")
    return data.copy()

# TODO use of other tokenizers ?

def getProcessedTokens(tF):
    '''
    Preprocesses the input text by removing tags, digits, stopwords, short text and stems
    Parameters : tF (Series) the input text feature
    '''
    CUSTOM_FILTERS = [ strip_tags, strip_punctuation
                    , strip_numeric, strip_non_alphanum
                    , remove_stopwords
                    , strip_multiple_whitespaces
                    # , stem_text
                    , lambda x: strip_short(x, minsize=3)
                    ]
    return [preprocess_string(sentence, CUSTOM_FILTERS) for sentence in tF]


def generateEmbeddings(tokens, keyedVectors):

    def sent_vec(sent):
        vector_size = keyedVectors.vector_size
        wv_res = np.zeros(vector_size)
        ctr = 1
        for w in sent:
            if w in keyedVectors:
                ctr += 1
                wv_res += keyedVectors[w]
        wv_res = wv_res/ctr
        return wv_res
    
    embeddings = tokens.apply(sent_vec)
    return embeddings


if __name__ == "__main__":

    usePretrained = False

    df = load_data('imdb_dataset.csv')
    textFeature = df['review']
    df['tokens'] = getProcessedTokens(textFeature)

    le = LabelEncoder()
    y= le.fit_transform(df['sentiment'])

    # this will decide the way the training and test data is split
    if usePretrained: # MODE 2 using a pre-trained embedding model
        # models are downloaded to the home directory cd ~
        keyedVectors = api.load('word2vec-google-news-300')
        df['embeddings'] = generateEmbeddings(df['tokens'], keyedVectors)

        X = df['embeddings'].to_list()
        # split can come after generating embeddings because we are not training the embedding model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # MODE 1 # training ur own embedding model 
    else:
        X = df['tokens']

        # This makes sure our embeddings model is trained only on the train set and does not cause target leakage by "seing" the test set while embedding
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 1st split * * * * * * *
        my_w2v = Word2Vec(sentences=X_train, vector_size=300, window=5, min_count=5, workers=4) # then train
        keyedVectors = my_w2v.wv

        X_train_e = generateEmbeddings(X_train, keyedVectors) # then embed train
        X_test_e = generateEmbeddings(X_test, keyedVectors) # then embed test

        X_train = X_train_e.to_list()
        X_test = X_test_e.to_list()

    
    classifier = LogisticRegression(solver = 'sag')
    classifier.fit(X_train,y_train)
    predicted = classifier.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, predicted))
    print("recall:",metrics.recall_score(y_test, predicted))

# doc 2 vec 
# tag = train.head().reset_index().apply(lambda x: TaggedDocument(x['tokens'], tags=[x.name]), axis=1)