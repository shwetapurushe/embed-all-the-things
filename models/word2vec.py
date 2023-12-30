# from gensim.test.utils import common_texts
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_tags, strip_numeric, strip_punctuation, strip_non_alphanum, strip_multiple_whitespaces, strip_short, stem_text
import pandas as pd
import gensim.downloader as api
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

 # getting sentence embeddings from word embeddings
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
    dataPath = os.path.join(os.getcwd(), dataset)
    data = pd.read_csv(dataPath)
# data = pd.read_csv("/Users/shweta/Documents/embedding-all-the-things/imdb_dataset.csv")
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
                    , stem_text
                    , lambda x: strip_short(x, minsize=2)
                    ]
    return [preprocess_string(sentence, CUSTOM_FILTERS) for sentence in tF]


def generateEmbeddings(tokens, pretrained=True, modelName = 'word2vec-google-news-300'):

    def sent_vec(sent):
        vector_size = word2vec_pretrained.vector_size
        wv_res = np.zeros(vector_size)
        ctr = 1
        for w in sent:
            if w in word2vec_pretrained:
                ctr += 1
                wv_res += word2vec_pretrained[w]
        wv_res = wv_res/ctr
        return wv_res
    
    # MODE 2 using a pre-trained embedding model
    if pretrained:
        word2vec_pretrained = api.load(modelName)
        embeddings = tokens.apply(sent_vec)
    # MODE 1 # training ur own embedding model 
    else:
        # train embedding model here
        pass
    
    return embeddings


if __name__ == "__main__":
    df = load_data('imdb_dataset.csv')
    textFeature = df['review']
    df['tokens'] = getProcessedTokens(textFeature)
    df['embeddings'] = generateEmbeddings(df['tokens'], pretrained=True, modelName='word2vec-google-news-300')

    le = LabelEncoder()
    X = df['embeddings'].to_list()
    y= le.fit_transform(df['sentiment'])
    # split can come after generating embeddings because we are not training the embedding model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = LogisticRegression(solver = 'sag')
    classifier.fit(X_train,y_train)
    predicted = classifier.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, predicted))
    print("recall:",metrics.recall_score(y_test, predicted))

