from transformers import BertTokenizer, RobertaTokenizer
import sys
from megasplittor2000 import split_into_lists_of_words
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from operator import itemgetter


#Script used to create a custom tokenizer (based on tokens found in our embedding corpus & dataset articles). 

def chunks(liste, n):
    for i in range(0, len(liste), n):
        yield liste[i:i+n]
    yield liste[len(liste)//n*n:]
#Open articles
PATH = Path("../storage/dataset_2/treated_articles")
iterd = PATH.iterdir()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', max_length = 512)
text_art = ''
for label in tqdm(iterd):
    for article in tqdm(label.iterdir()):
        text_art += open(str(article), 'r', encoding='utf-8').read().replace('\n\n', '. ').replace('..', '.')

text = open('../storage/cybersec_final.txt', 'r', encoding='utf-8').read()
#process corpus then split into list of words. 
text = text.replace('\n\n', '. ')
text = text.replace('..', '.')
text = text.replace('#', '')
text_art = text_art.replace('\n\n', '. ')
text_art = text_art.replace('..', '.')
text_art = text_art.replace('#', '')
data_art = split_into_lists_of_words(text_art)
data_emb = split_into_lists_of_words(text)
data_art = [x for sent in data_art for x in sent]
data_emb = [x for sent in data_emb for x in sent]

#Get the 15000 most common words from the embedding corpus, every token appearing more than 5 times in the dataset.
c = Counter(data_art)
d = Counter(data_emb)
data_art =  set(x for x in c.keys() if c[x] > 5)
data_emb = set(map(itemgetter(0), d.most_common(15000)))
print(len(tokenizer))
data = list(data_art | data_emb)
print(len(data), len(tokenizer))
for x in tqdm(chunks(data, 1000)):
    tokenizer.add_tokens(x)
tokenizer.save_pretrained('../storage/tokenizer')