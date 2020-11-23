import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
plt.style.use('ggplot')

#set filepath
os.chdir('/home/ojas/Documents/independent_study_ed_geheringer')

#filename = "suggestions_expertiza_fall2018.csv"
filename = "problems_expertiza_fall2018.csv"

df = pd.read_csv(filename, encoding = "ISO-8859-1")
sentences = df['Comments'].values
#df['Suggest_Solutions'] = df['Suggest_Solutions'].replace([-1], 0)
df['Mentions_Problem'] = df['Mentions_Problem'].replace([-1], 0)
#values = df["Suggest_Solutions"].values
values = df["Mentions_Problem"].values
sentences_train, sentences_test, y_train, y_test = train_test_split(
                    sentences, values, test_size=0.15, random_state=1000)

#Word indexing based on frequency
tokenizer = Tokenizer(num_words=5000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ',
                       char_level=False, oov_token=None)

tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

#Create Word embeddings using pretrained GloVe model
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

embedding_dim = 50

os.chdir("/home/ojas/Documents/independent_study_ed_geheringer/glove.6B")
embedding_matrix = create_embedding_matrix(
        'glove.6B.50d.txt',
        tokenizer.word_index, embedding_dim)
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(10, activation='relu'))

# Hidden - Layers
model.add(layers.Dropout(0.6, noise_shape=None, seed=None))
model.add(layers.Dense(5, activation = "relu"))
model.add(layers.Dropout(0.6, noise_shape=None, seed=None))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

plot_history(history)