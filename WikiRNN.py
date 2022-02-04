#!/usr/bin/env python
# coding: utf-8

# Matthias J. Moser
# 00931524
# Jan Moser
# 51842006
# Thomas Kristan
# 51841768
# Michael Pölzl
# 51811422

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, Flatten
from keras.models import Model, Sequential, load_model
from keras import initializers, regularizers, constraints, optimizers, layers
#import kerastuner as kt
import seaborn as sns

df_train_csv = pd.read_csv('train.csv')
df_test_csv = pd.read_csv('test.csv')
df_test_labels_csv = pd.read_csv('test_labels.csv')


# <----- inspect data ------

print(df_train_csv.head())
print(df_test_csv.head())
print('-> test data without labels \n')



df_train = df_train_csv[['comment_text']]
df_label = df_train_csv.drop(columns=['comment_text', 'id']).fillna(0)
print(df_label.head(100))
df_test = df_test_csv[['comment_text']]
df_test_labels = df_test_labels_csv.drop(columns=['id']).fillna(0)

print(df_train.isnull().any())

print(df_label.value_counts())
print( '''->This means multi label classification
(loss: binary_crossentropy, activation: sigmoid)
''')



# df_label['non_toxic'] = (~df_label.sum(axis=1).astype('bool')).astype(int) -> nur für single label classsification

print(df_train['comment_text'].str.len().describe())
print( '->Avg 400 character per comment\n')


print(df_label[df_train['comment_text'].str.contains('\*')].value_counts())
print('-> some special characters !*#%$ are often used to censor \'bad\' words -> so we dont filter them')


# ----- inspect data ------>


# <----- prepare data ------
# define max words, only most common ones will be kept (10000 different words)
max_features=10000

# first sept of pre-processing: lower the words since words with capital letters would be treated 
# different in the model (other position in the feature vectos).
# Eg 'Nice' would be in a slighlty different location than 'nice' but they are semantically the same (same meaning)
words_train=df_train['comment_text'].str.lower()
words_test=df_test['comment_text'].str.lower()


# vectorize a text corpus
# apply filters on the words to eliminate special characters that do not carry any semantics relevant for classification.
tokenizer= Tokenizer(
    filters='"%&()+,-./:;<=>?@[\\]^_`{|}~\t\n',
    num_words=max_features,
    lower= True)

# apply tokenizer on comments
tokenizer.fit_on_texts(list(words_train))
#converts text into sequence of integers, each number shows how many times the word occured
tokenized_train=tokenizer.texts_to_sequences(words_train) 
tokenizer.fit_on_texts(list(words_test))
tokenized_test=tokenizer.texts_to_sequences(words_test)




print(words_test[0])
print(tokenized_test[0])

num_words = [len(c) for c in tokenized_train]
sns.histplot(data=num_words, bins=np.arange(0,250,10))
plt.xlabel('number of words')
plt.ylabel("count of comments")
plt.title('How many words per comment?')
plt.show()
print('-> max 100 words necessary')

max_len=100 

# create equal sized  2D Numpy arrays 
train_x=pad_sequences(tokenized_train,maxlen=max_len) 
test_x=pad_sequences(tokenized_test,maxlen=max_len)


# deleting rows labeled with value -1
mask = df_test_labels['toxic'] >= 0
test_labels = df_test_labels[mask]
test_X = test_x[mask]


# ----- prepare data ------>



# <----- model ------

def plot_history(his, model_name='model', trim=3):
    loss = pd.DataFrame(his.history['loss'], index=np.arange(1, len(his.history['loss'])+1), columns=['train'])
    if 'val_loss' in his.history:
        loss['val'] = his.history['val_loss']
    loss=loss.iloc[trim:]

    acc = pd.DataFrame(his.history['accuracy'], index=np.arange(1, len(his.history['accuracy'])+1), columns=['train'])
    if 'val_accuracy' in his.history:
        acc['val'] = his.history['val_accuracy']
    acc=acc.iloc[trim:]

    hisplt, axes = plt.subplots(1, 2, figsize=(15,5))
    hisplt.suptitle('Training History "{}"'.format(model_name))

    sns.lineplot(ax=axes[0], data=loss)
    axes[0].set(title='Loss', xlabel='Epochs', ylabel='Loss Value')
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    sns.lineplot(ax=axes[1], data=acc)
    axes[1].set(title='Accuracy', xlabel='Epochs', ylabel='Accuracy')
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


epochs = 15
batch_size = 64

def model_simple():
    # without embedded
    global train_x
    global df_label

    train_x = train_x.reshape(-1, 1, max_len)
    df_label = np.array(df_label).reshape(-1, 1, 6)


    model =  tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        64, return_sequences=True, input_shape=(1,100)))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(6,activation='sigmoid'))

    model.compile(
            loss='binary_crossentropy',
            optimizer= tf.keras.optimizers.Adam(lr=1e-3),
            metrics=['accuracy'])
    print(model.summary())
    return model

def model_embedding_1():
    model = Sequential()
    model.add(Embedding(20000, 64))
    model.add(LSTM(30, return_sequences=True))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.1))
    model.add(Dense(25, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation="sigmoid"))

    model.compile(
            loss='binary_crossentropy',
            optimizer= tf.keras.optimizers.Adam(lr=1e-3),
            metrics=['accuracy'])

    print(model.summary())
    return model

def model_bi():
    model = Sequential()
    model.add(Embedding(20000, 256))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.1))
    model.add(GlobalMaxPool1D())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation="sigmoid"))

    model.compile(
            loss='binary_crossentropy',
            optimizer= tf.keras.optimizers.Adam(lr=1e-3),
            metrics=['accuracy'])

    print(model.summary())
    return model


def model_gru():
    model = Sequential()
    model.add(Embedding(20000, 64))
    model.add(GRU(224, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GRU(480, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GRU(256, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GlobalMaxPool1D())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation="sigmoid"))

    model.compile(
            loss='binary_crossentropy',
            optimizer= tf.keras.optimizers.Adam(lr=1e-3),
            metrics=['accuracy'])

    print(model.summary())
    return model

# using the best model and fine tune it using the keras-tuner framework
def model_gru_tuned(hp):
    model = Sequential()
    model.add(Embedding(20000, 64))

     # Choose an optimal value between 32-512
    hp_units = hp.Int('units_1', min_value=32, max_value=512, step=32)
    model.add(GRU(units=hp_units, return_sequences=True))
    model.add(Dropout(hp.Choice('dropout_1', values=[0.1,0.2,0.3,0.4])))
    hp_units_1 = hp.Int('units_2', min_value=32, max_value=512, step=32)
    model.add(GRU(hp_units_1, return_sequences=True))
    model.add(Dropout(hp.Choice('dropout_2', values=[0.1,0.2,0.3,0.4])))
    hp_units_2 = hp.Int('units_3', min_value=64, max_value=512, step=32)
    model.add(GRU(hp_units_2, return_sequences=True))
    model.add(Dropout(hp.Choice('dropout_3', values=[0.1,0.2,0.3,0.4])))
    model.add(GlobalMaxPool1D())
    hp_units_3 = hp.Int('units_4', min_value=128, max_value=512, step=64)
    model.add(Dense(hp_units_3, activation="relu"))
    model.add(Dropout(hp.Choice('dropout_4', values=[0.1,0.2,0.3,0.4])))
    model.add(Dense(6, activation="sigmoid"))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
            loss='binary_crossentropy',
            optimizer= tf.keras.optimizers.Adam(lr=hp_learning_rate),
            metrics=['accuracy'])

    print(model.summary())
    return model

# <----- choose model ------

callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',patience=2,),
        ]

model_val = model_gru() # best model used

history = model_val.fit(
        train_x,
        df_label,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks_list,
        verbose=1
        )

#model_val.save("model_gru")
plot_history(history, trim=0)

# ----- choose model ------>

# <----- hyper parameter tuning ------
# uncomment to start hyper parameter tuning
""""
# <----- hyper parameter tunining ------

tuner = kt.Hyperband(model_gru_tuned, # the hypermodel
                     objective='val_accuracy', # objective to optimize
                     max_epochs=10,
                     factor=3, # factor which you have seen above 
                     directory='.', # directory to save logs 
                     project_name='khyperband')
tuner.search_space_summary() 

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# Perform hypertuning
tuner.search(train_x, df_label, epochs=10, validation_split=0.2, callbacks=[stop_early])
# ----- hyper parameter tunining ------>
"""
# ----- hyper parameter tuning ------>


# <----- validation ------
model_test = model_gru()

history = model_test.fit(
        train_x,
        df_label, 
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        validation_data=(test_X, test_labels),
        verbose=1
        )

plot_history(history, trim=0)
# ----- validation ------>

#predicting trump tweets
model_pred = load_model("model_final_test")

df_trump = pd.read_csv('trump.csv')
words_trump=df_trump['comment_text'].str.lower()
tokenizer.fit_on_texts(list(words_trump))
tokenized_trump=tokenizer.texts_to_sequences(words_trump)
pred_trump=pad_sequences(tokenized_trump,maxlen=max_len)

pred = model_pred.predict(pred_trump)