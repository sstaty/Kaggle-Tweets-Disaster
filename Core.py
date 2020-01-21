import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import spellchecker
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#stop = set(stopwords.words('english'))

# Data visualisation

tweet = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
print(tweet.head(3))

x = tweet.target.value_counts()
print(x)
sns.barplot(x.index, x)
plt.gca().set_ylabel('samples')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
tweet_len = tweet[tweet['target'] == 1]['text'].str.len()
ax1.hist(tweet_len, color='red')
ax1.set_title('disaster tweets')
tweet_len = tweet[tweet['target'] == 0]['text'].str.len()
ax2.hist(tweet_len, color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
plt.show()

# Data pre-processing

df = pd.concat([tweet, test])
print(df.shape)

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

df['text'] = df['text'].apply(lambda x : remove_URL(x))

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'', text)

df['text']=df['text'].apply(lambda x : remove_html(x))

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

df['text'] = df['text'].apply(lambda x: remove_emoji(x))

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


spell = spellchecker.SpellChecker()

def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

df['text'] = df['text'].apply(lambda x : correct_spellings(x))


# Glove Representation

#def create_corpus(df):
#    corpus = []
#    for tweet_curr in tqdm(df['text']):
#        words = [word.lower() for word in word_tokenize(tweet_curr) if (word.isalpha() == 1)]
#        corpus.append(words)
#    return corpus

#corpus=create_corpus(df)

for i in tqdm(range(1000000)):
    pass









