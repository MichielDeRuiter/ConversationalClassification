# %%
import os
import re
import csv
import string
import random
import pickle
import nltk
import pandas as pd
import itertools
import collections
import matplotlib.pyplot as plt
from nltk.corpus import reuters, movie_reviews  # reuters (news), brown, movie_reviews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# sid = SentimentIntensityAnalyzer()
# nltk.download('vader_lexicon')
# nltk.download('reuters')
# nltk.download('movie_reviews')

# %%
# Dataset: https://www.kaggle.com/thoughtvector/customer-support-on-twitter
twitter = pd.read_csv('twitter_30k.csv')
twitter = twitter.loc[twitter['inbound'] == False]  # Only use responses from companies
twitter = twitter.reset_index()
twitter = twitter[['text']]
twitter = twitter.dropna(how='all', axis=0)

# df
# df['text']
# print(df.shape[0])
# df[df['text'] == ''].index
# df.columns[df.isnull().any()]
# df = df.head(1000) # TEMPORARY! # TEMPORARY!# TEMPORARY!# TEMPORARY!# TEMPORARY!# TEMPORARY!# TEMPORARY!# TEMPORARY!# TEMPORARY!

# %%
# Dataset: http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/

rootdir = 'ubuntu_dialogs/ubuntu_dialogs/dialogs/'
ubuntu = None

for subdir, dirs, files in os.walk(rootdir):
    for file in files:

        filepath = subdir + os.sep + file
        data = pd.read_csv(filepath, error_bad_lines=False, sep='\t', quoting=csv.QUOTE_NONE, header=None)
        data.drop(data.columns[0:3], axis = 1, inplace = True)

        # TODO: 10k vervangen door cutoff
        ubuntu = data if ubuntu is None else ubuntu.append(data)
        if len(ubuntu) >= 500:
            break

ubuntu.rename(columns={ ubuntu.columns[0]: "text" }, inplace = True)
print(ubuntu)


# %%
# print(twitter)
# print(twitter.shape[0])

#%%

# %%
# assign labels
pos = []
neg = []
n = 10000

for i in range(n):
    pos.append((twitter['text'].iloc[i], 1))
    pos.append((ubuntu['text'].iloc[i], 1))
    # TOEVOEGEN: pos.append(Ubuntu data)

i = 0
for fid in reuters.fileids():
    neg.append((' '.join(word for word in reuters.words(fid)), 0))
    i += 1
    if i == n:
        break
    
i = 0
for fid in movie_reviews.fileids():
    neg.append((' '.join(word for word in movie_reviews.words(fid)), 0))
    i += 1
    if i == n:
        break

# %%
# Balance the amount of examples and shuffle the data.
# cutoff = len(neg) if len(neg) <= len(pos) else len(pos)
# pos = pos[:cutoff]
# neg = neg[:cutoff]
data = pos + neg
random.shuffle(data)

# %%
# Clean data
Stemmer = SnowballStemmer('english')
Lemmatizer = WordNetLemmatizer()

def clean_data(data):
    columns = ['text', 'label']

    # array of tuples to df
    df = pd.DataFrame(data, columns = columns) 

    # Delete non-ASCII
    printable = set(string.printable)
    df['text'] = df['text'].apply(lambda y: ''.join(filter(lambda x: x in printable, y)))

    # Delete weblinks
    df['text'] = df['text'].apply(lambda x: re.sub(
        r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', x, flags=re.MULTILINE))

    # Delete words containing special characters, except standard text characters.
    punctuation = set(string.punctuation)
    # include = ['(', ')', ',', '.', '!', '?']
    include = ['(', ')', '!', '?']
    exclude_punctuation = [p for p in punctuation if p not in include]
    for e in exclude_punctuation:
        regex_p = '\\' + e + '[A-Za-z0-9]+'
        try:
            df['text'] = df['text'].apply(
                lambda x: re.sub(r'{}'.format(regex_p), '', str(x)))
        except:
            pass

    # Delete special characters
    for e in exclude_punctuation:
        df['text'] = df['text'].str.replace(e, '')
        
    # Delete single characters
    df['text'] = df['text'].apply(lambda x: re.sub(r'(?:^| )\w(?:$| )', '', x).strip())

    # Delete words containing numbers
    df['text'] = df['text'].apply(lambda x: re.sub(r'\w*\d\w*', '', x).strip())

    # To lowercase
    df['text'] = df['text'].apply(lambda x: x.lower())

    # Strings to bag of words
    df['text'] = df['text'].str.split()

    # Stemming
    # df['text'] = df['text'].apply(lambda x: [Stemmer.stem(w) for w in x])

    # Lemmatizing
    # df['text'] = df['text'].apply(lambda x: [Lemmatizer.lemmatize(w) for w in x])

    # df to array of tuples
    data = [tuple(x) for x in df[columns].to_numpy()] 

    return data


# %%
data = clean_data(data)
training_percentage = 0.8
total_rows = len(data)
training_amount = int(training_percentage * total_rows)
train = data[:training_amount]
test = data[training_amount:]
n_gram = 3
# n_gram = None


# %%
# Get the separate words in text.
def get_words_in_text(text):
    all_words = []
    for (words, sentiment) in text:
        all_words.extend(words)
    return all_words

# Construct our features based on which text contain which word.
def extract_features(document):
    word_features = nltk.FreqDist(get_words_in_text(train))
    document_words = set(document)
    features = {}
    if n_gram == None:
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
    else: 
        ngram_vocab = nltk.ngrams(document, n_gram)
        features = dict([(ng, True) for ng in ngram_vocab])
    return features


# %%
train_set = nltk.classify.apply_features(extract_features, train)
test_set = nltk.classify.apply_features(extract_features, test)

# %%
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(classifier.show_most_informative_features(32))
print('accuracy:', nltk.classify.util.accuracy(classifier, test_set))



# %%


# %%
# twitter['text'] = twitter['text'].str.split()
# words_in_tweet = twitter['text']

# # %%
# print(type(twitter['text'].iloc[i]))
# print(type(df['text'].iloc[i]))

# # %%
# # List of all words across tweets
# all_words_no_urls = list(itertools.chain(*words_in_tweet))

# # Create counter
# counts_no_urls = collections.Counter(all_words_no_urls)

# counts_no_urls.most_common(15)
# [('climate', 550),
#  ('the', 528),
#  ('change', 395),
#  ('to', 355),
#  ('and', 227),
#  ('of', 227),
#  ('a', 200),
#  ('is', 180),
#  ('in', 168),
#  ('for', 133),
#  ('we', 112),
#  ('coronavirus', 99),
#  ('on', 98),
#  ('how', 72),
#  ('climatechange', 71)]

# # %%
# clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
#                              columns=['words', 'count'])

# clean_tweets_no_urls.head()

# # %%
# fig, ax = plt.subplots(figsize=(8, 8))

# # Plot horizontal bar graph
# clean_tweets_no_urls.sort_values(by='count').plot.barh(x='words',
#                       y='count',
#                       ax=ax,
#                       color="purple")

# ax.set_title("Most common Words Found in customer support Twitter responses")

# plt.show()

#     # %%
# df


# %%



# %%
### Prepare Dataset for Overfitting Test or RNN.
pos_train = []
neg_train = []
pos_test = []
neg_test = []

# n= 100
# n = 800
# n = 1600
n = 4000
seed = 1

for i in range(n):
    pos_train.append((twitter['text'].iloc[i], 1))
    pos_test.append((ubuntu['text'].iloc[i], 1))
    # TOEVOEGEN: pos.append(Ubuntu data)

i = 0
for fid in reuters.fileids():
    neg_train.append((' '.join(word for word in reuters.words(fid)), 0))
    i += 1
    if i == n:
        break

i = 0
for fid in movie_reviews.fileids():
    neg_test.append((' '.join(word for word in movie_reviews.words(fid)), 0))
    i += 1
    if i == n:
        break

pos_test = pos_test[:int(n/4)]
neg_test = neg_test[:int(n/4)]

train = pos_train + neg_train
test = pos_test + neg_test

train = clean_data(train)
test = clean_data(test)

random.Random(seed).shuffle(train)
random.Random(seed).shuffle(test)


#%%
### OVERFITTING TEST: Training on Twitter and Reuters, testing on ubuntu and movie_reviews ###
train_set = nltk.classify.apply_features(extract_features, train)
test_set = nltk.classify.apply_features(extract_features, test)

classifier = nltk.NaiveBayesClassifier.train(train_set)

print(classifier.show_most_informative_features(32))

print('accuracy:', nltk.classify.util.accuracy(classifier, test_set))

#%%
# Save model
f = open('ConversationalClassifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()



# %%
### Get most informative features of Reuters vs Twitter ###
# data = []
# n = 800

# n_gram = 1

# for i in range(n):
#     data.append((twitter['text'].iloc[i], 1))
#     data.append((ubuntu['text'].iloc[i], 0))
#     # TOEVOEGEN: pos.append(Ubuntu data)

# training_percentage = 0.8
# total_rows = len(data)
# training_amount = int(training_percentage * total_rows)

# train = data[:training_amount]
# test = test = data[training_amount:]

# train = clean_data(train)
# test = clean_data(test)

# train_set = nltk.classify.apply_features(extract_features, train)
# test_set = nltk.classify.apply_features(extract_features, test)

# classifier = nltk.NaiveBayesClassifier.train(train_set)

# print(classifier.show_most_informative_features(32))

# print('accuracy:', nltk.classify.util.accuracy(classifier, test_set))


# %%
#### LSTM ####

# print(type(df.0))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
%matplotlib inline


# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)
tf.random.set_random_seed(seed_value) 
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# for later versions:
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)


# le = LabelEncoder()
# Y = le.fit_transform(Y)
# Y = Y.reshape(-1,1)

# X = df.text
# Y = df.label
# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

columns = ['text', 'label']
train_df = pd.DataFrame(train, columns = columns)
test_df = pd.DataFrame(test, columns = columns)

X_train = train_df.text
X_test = test_df.text
Y_train = train_df.label
Y_test = test_df.label

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.25,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

# %%
# X_test

# # %%
# Y_test

# %%
model.save('Keras_LSTM_Classifier_873accuracy_18000train_1000validate_1000test.h5')


# %%
# from keras.models import load_model
# model = load_model('Keras_LSTM_Classifier_1600train_400test.h5')

# %%
# twitter_30k = pd.read_csv('twitter_customer_support.csv')
# twitter_30k = twitter_30k.head(30000)
# twitter_30k.to_csv('twitter_30k.csv')

# %%
