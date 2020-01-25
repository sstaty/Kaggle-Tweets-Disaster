import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
#import spellchecker
import os
from tqdm import tqdm
import tweetcleaner
import gloveconv
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
#plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
tweet_len = tweet[tweet['target'] == 1]['text'].str.len()
ax1.hist(tweet_len, color='red')
ax1.set_title('disaster tweets')
tweet_len = tweet[tweet['target'] == 0]['text'].str.len()
ax2.hist(tweet_len, color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
#plt.show()

# Data pre-processing

df = pd.concat([tweet, test])
print(df.shape)

df_text = df['text']
print(type(df_text))
print(type(tweet))

df['text'] = df['text'].apply(lambda x: tweetcleaner.remove_url(x))
df['text'] = df['text'].apply(lambda x: tweetcleaner.remove_html(x))
df['text'] = df['text'].apply(lambda x: tweetcleaner.remove_emoji(x))


# Glove Representation

corpus = gloveconv.create_corpus(df)
embedding_dict = gloveconv.load_glove('../Glove/glove.6B.100d.txt')








