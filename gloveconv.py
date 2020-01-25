import tqdm
import nltk
import numpy as np


def create_corpus(df):
    corpus = []
    for tweet_curr in tqdm.tqdm(df['text']):
        words = [word.lower() for word in nltk.tokenize.word_tokenize(tweet_curr) if (word.isalpha() == 1)]
        corpus.append(words)
    return corpus


def load_glove(path):
    embedding_dict = {}
    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            embedding_dict[word] = vector
    f.close()
    return embedding_dict
