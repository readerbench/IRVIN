from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
import contractions
import os
from pathlib import Path
from tqdm import tqdm
import re 


#Script used to remove actions, spaces, symbols, IP addresses, links etc... from labeled articles. 
#All there's left is letters, and the symbols we chose to keep.

kept = [',', '.', '"', ':', ')', '(','!', '?' ';']
puncts = ['-','|', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', ',', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
PATH = Path("../storage/dataset_2/articles_2")
iterd = PATH.iterdir()
for label in iterd:#on peut bien modifier iterdir c'est pas grave
    for article in tqdm(label.iterdir()):
        list_lines = []
        with open(str(article), 'r', encoding = 'utf-8') as file :
            for line in file : 
                new_line = unidecode.unidecode(line)#remove accents
                for el in kept : 
                    if el in new_line :  
                        new_line = new_line.replace(el,' '+el+' ')#add spaees before and after characters we want to keep, no worries if there is too much because next function fixes it. 
                new_line = contractions.fix(new_line)#change n't to not and 're to are
                new_line = re.sub(r'http\S+', '', new_line)#remove urls

                for punct in puncts :
                    if punct in new_line : 
                        new_line = new_line.replace(punct, '')#remove useless punct
                if bool(re.search(r'\d', new_line)):
                    new_line = re.sub(r'\d', '#', new_line)
                    new_line = new_line.replace("#,#", '#')
                    new_line = new_line.replace("#.#","#")
                    new_line = new_line.replace("#,#", '#')
                    new_line = new_line.replace("#.#","#")
                    new_line = new_line.replace("#,#", '#')
                    new_line = new_line.replace("#.#","#")    
                new_line = new_line.strip()
                new_line = " ".join(new_line.split())#remove useless whitespaces   
                list_lines.append(new_line + '\n')
            print(str(article))
            name = str(article).split('/')[-2:]
            print(name)

        with open("../storage/dataset_2/treated_articles/"+name[0]+'_TREATED'+'/'+name[1], 'w', encoding = 'utf-8') as g : 
            g.writelines(list_lines)

