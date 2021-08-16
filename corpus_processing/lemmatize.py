import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import fileinput
from tqdm import tqdm
from pathlib import Path


# This script allows to lemmatize the articles from the dataset (only used to test the Multinomial Naive Bayes baseline)


PATH = Path("../storage/dataset_2/dataset")
lemmatizer = WordNetLemmatizer()
iterd = PATH.iterdir()
for dataset in iterd:
    for file in tqdm(dataset.iterdir()):
        f = open(str(file), 'r', encoding = 'utf-8')
        name = str(file).split('/')[-2:]
        g = open(f'../storage/dataset_2/lemmatized_dataset/{name[0]}/{name[1]}', 'w', encoding='utf-8')
        print(name)
        for line in f:
            line_2 = ' '.join(
                [lemmatizer.lemmatize(w) if w not in ["has", 'was'] else w for w in line.rstrip().split()]
            )
            
            print(line_2, file = g)
        
        g.close()
        f.close()
            # overwrites current `line` in file
            