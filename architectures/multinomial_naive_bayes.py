from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import sys
from eval import Performance, show_top, show_top10
import numpy as np
import pandas as pd


#Little work on MNB to test the ability of a naive classifier to classify articles. 
#Gridsearch on the size of the vocabulary kept (limited to 3000 because the dataset has only 1600 articles) and type of embedding (TF-IDF or count). 
liste_sizes = np.arange(50, 3000, 50)
list_conf_mat = []
if __name__ == '__main__':
    #Prepare data
    store = pd.HDFStore("../storage/storage_embeddings_2.h5")
    df_train = store["df_counts_train"]
    y_train = df_train["labels"].to_numpy()
    df_test = store["df_counts_test"]
    y_test = df_test["labels"].to_numpy()
    list_col_init = [x for x in df_train.columns if (np.count_nonzero(df_train[x]) > 10 and x != "labels")]
    X_train = df_train[list_col_init]
    X_test = df_test[list_col_init]
    #Train the architecture to see the least relevant words to keep. 
    MNB = MultinomialNB()
    MNB.fit(X_train, y_train)
    topwords = show_top(MNB, list_col_init)
    for i, size in enumerate(liste_sizes) :
        list_col = topwords[-size:]
        print(list_col)
        X_train = df_train[list_col]
        X_test = df_test[list_col]
        #Train the architecture with the given voc size. 
        MNB = MultinomialNB()
        MNB.fit(X_train, y_train)
        y_pred = MNB.predict(X_test)
        #print the top 10 words in articles classified relevant. 
        print(show_top10(MNB, list_col))
        perf = Performance()
        list_conf_mat.append(perf.confusion_matrix(y_pred, y_test))
    df = pd.DataFrame(np.hstack(list_conf_mat), columns = np.repeat(liste_sizes, 2))
    df.to_csv('../results/confmats_counts.csv')
