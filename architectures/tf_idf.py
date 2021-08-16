from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import csv


#Script used to embed texts using TF-IDF. 


nltk.download('stopwords')
list_stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

PATH = Path("../storage/dataset_2/lemmatized_dataset")
def tf_idf(path):
    filenames = []
    labels = []
    iterd = PATH.iterdir()
    for cat in iterd : 
        filenames.append([str(x) for x in cat.iterdir()])
        labels.append([('IRRELEVANT_TREATED' not in str(x)) for x in cat.iterdir()])
        print(labels)
    x_train, x_test, y_train, y_test = filenames[0], filenames[1], labels[0], labels[1]
    #Generate a file with the filenames in the right order.
    result_file = open("../storage/filenames_tfidf.csv",'w')
    wr = csv.writer(result_file, dialect='excel')
    for file in x_train :
        wr.writerow(file)
    for file in x_test: 
        wr.writerow(file)
    #Vectorize articles, then store it. 
    vectorizer = TfidfVectorizer(input = 'filename', stop_words = list_stopwords)
    vectors = vectorizer.fit_transform(x_train)
    vectors_test = vectorizer.transform(x_test)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    dense_test = vectors_test.todense()
    denselist = dense.tolist()
    denselist_test = dense_test.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    df_test = pd.DataFrame(denselist_test, columns=feature_names)
    df["labels"] = y_train
    df_test["labels"] = y_test
    store = pd.HDFStore("../storage/storage_embeddings_2.h5")
    store["df_tfidf_train"] = df
    store["df_tfidf_test"] = df_test


if __name__ == "__main__":
    tf_idf(PATH)
    store = pd.HDFStore("../storage/storage_embeddings.h5")
    df = store["df_tfidf_train"]
    list_col = [x for x in df.columns if np.count_nonzero(df[x]) > 1]
    print(list_col)