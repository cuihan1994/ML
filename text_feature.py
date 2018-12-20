import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos,neg):
    lexicon = []
    for fi in [pos,neg]:
        with open(fi,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]

    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        if 1000>w_counts[w]>20:
            l2.append(w)
    print(len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []
    with open(sample,'r')as f:
        contents = f.readlines()
        for l in contents[]
create_lexicon('pos.txt','neg.txt')