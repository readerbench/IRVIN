import numpy as np
import re 
from tqdm import tqdm
import re 
import fasttext


#This script corresponds to the first processing applied on corpus from the different websites we scrapped
#The text processing changes for each website, I put 2 examples below



filename = "../storage/embedding/brut/ehackernews.txt"
new_file = "../storage/embedding/treated/ehackernews_treated.txt"
f = open(filename, "r")
g = open(new_file, 'a')


"""welivesecurity
PRETRAINED_MODEL_PATH = 'lid.176.bin'#à télécharger
model = fasttext.load_model(PRETRAINED_MODEL_PATH)
for line in tqdm(f) :
    if line != "": 
        if len(line) < 90 or model.predict([line[:-2]])[0][0][0] != '__label__en': 
            continue
        new_line = line
        idx = 0
        while new_line[idx] == ' ':
            idx += 1
        new_line = new_line[idx:]
        if new_line[0] == 'by':
            tab = new_line.split(' ', 3)#liste des 3 premiers mots et du reste
            if len(tab[2]) == 1 : 
                tab = tab[-1].split(' ', 1)#liste du 'vrai nom' et de la suite
            new_line = tab[-1]
        if new_line[0:25].lower() == 'sophos experts learn more' : 
            new_line = new_line[26:]
        print(new_line, file = g)
f.close()
g.close()
"""
#ehackernews
for line in tqdm(f) :
    if line != "": 
        if len(line) < 90 or 'cooki' in line or '|' in line or line[0:4] == 'http'or '@' in line or '[' in line:  
            continue
        new_line = line
        if new_line[1] == ")" or new_line[0] in ["•", '-', '*']:
            new_line = new_line[2:]
        if 'http/' in line :
            idx = new_line.find('http/')
            x = line[idx]
            while line[idx] != ' ':
                if line[idx] != '.' or line[idx+1] != ' ': 
                    idx += 1
                else : 
                    break
        print(new_line, file = g)
f.close()
g.close()