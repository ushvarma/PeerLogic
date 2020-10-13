import numpy as np
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
import os

os.environ['KERAS_BACKEND']='tensorflow' # Or theano
from keras import initializers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from keras.layers.core import Layer
from keras.models import model_from_json
from keras.models import load_model
import json
import matplotlib.pyplot as plt
from PIL import Image

reviews = pd.read_csv("suggestions_expertiza_fall2018_redone.csv") 
new_data = reviews.filter(["REVIEW","TAG"])
new_data = new_data.loc[new_data.REVIEW.apply(lambda x: not isinstance(x, (float, int)))]
train_set, test_set = train_test_split(new_data, test_size=0.05, random_state=42)
train_reviews = list(train_set["REVIEW"])
train_labels = list(train_set["TAG"])

for i in range(len(train_reviews)):
  train_reviews[i] = re.sub('\d',' ',train_reviews[i]) # Replacing digits by space
  train_reviews[i] = re.sub(r'\s+[a-z][\s$]', ' ',train_reviews[i]) # Removing single characters and spaces alongside
  train_reviews[i] = re.sub(r'\s+', ' ',train_reviews[i]) # Replacing more than one space with a single space

for i in range(len(train_reviews)):
    if 'www.' in train_reviews[i] or 'http:' in train_reviews[i] or 'https:' in train_reviews[i] or '.com' in train_reviews[i]:
        train_reviews[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_reviews[i])
        
# Training model only on a subset of data
from sklearn.utils import shuffle
from keras.preprocessing.text import text_to_word_sequence
x, y = shuffle(train_reviews, train_labels)

#Backup
train_x = train_reviews
train_y = train_labels

review_len = 0
sentence_len = 0

for review  in x:
    sentences = review.split(".")
    review_len += len(sentences)
    for sentence in sentences:
        sentence_len += len(text_to_word_sequence(sentence))

# Based on output set max sentence amd max sentense length variables below
print(int(review_len/ len(train_reviews)))
print(int(sentence_len / len(train_reviews)))

MAX_SENTENCES = 3 # Maximum number of sentences to be considered as a review for each sample
MAX_SENTENCE_LENGTH = 33 # Maximum number of words to be considered as a sentence for each sample
MAX_NB_WORDS = 800 # This specifies how many top tokens in each review to be stored
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 100 # Padding

# Takes 5 minutes to run on entire training dataset
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_reviews)
train_sequences = tokenizer.texts_to_sequences(train_reviews)

word_index = tokenizer.word_index
print('Number of Unique Tokens',len(word_index)) # Total 996497 unique words 

#Padding
#train_x = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
#type(train_x)

# !cp gdrive/My\ Drive/glove100d.zip .
# !unzip glove100d.zip

embeddings_index = {}
for i, line in enumerate(open('glove.6B.100d.txt')):
  values = line.split() # 0 th index will be the word and rest will the embedding vector (size 100 as we have used Glove.6B.100D embedding file) 
  embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

# create token(words in word index)-embedding mapping
embedding_matrix = np.zeros((len(word_index) + 1, 100)) # 100 since embedding_dimesion is 100, +1 because index 0 is reserved in word_index
for word, i in word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector
# We can initialize random vector and assign for words which are not present in embeddings.Other option is keep trainable=true in embedding layer of the NN model.
# We choose 2nd option

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
# nonzero_elements / len(word_index)

data = np.zeros((len(train_x), MAX_SENTENCES, MAX_SENTENCE_LENGTH))

for i, review in enumerate(train_x):
    sentences = review.split(".")
    for j, sentence in enumerate(sentences):
    # Number of sentences should be less than the maximum
        if j < MAX_SENTENCES:            
            wordTokens = text_to_word_sequence(sentence)
            k = 0
            for word in wordTokens:
                if k < MAX_SENTENCE_LENGTH and word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = word_index[word]
                    k += 1
train_x = data

# Attention Layer
def dot_product(x, kernel):

    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):

        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(a)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]

# Model architecture
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SENTENCE_LENGTH,trainable=True)

# Words level attention model
word_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(word_input)
word_lstm = Bidirectional(GRU(64, return_sequences=True,recurrent_dropout=0.5))(embedded_sequences)
word_att = Attention()(word_lstm)
word_drp = Dropout(0.4)(word_att)
wordEncoder = Model(word_input, word_drp)

# Sentence level attention model
sent_input = Input(shape=(MAX_SENTENCES, MAX_SENTENCE_LENGTH), dtype='int32')
sent_encoder = TimeDistributed(wordEncoder)(sent_input)
sent_lstm = Bidirectional(GRU(64, return_sequences=True, recurrent_dropout = 0.5))(sent_encoder)
sent_att = Attention()(sent_lstm)
sent_drp = Dropout(0.5)(sent_att)
preds = Dense(1, activation='sigmoid')(sent_drp)
model = Model(sent_input, preds)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.summary()

# Keeping a checkpoint to store only the model which gives best output validation accuracy
chkpt=ModelCheckpoint('expertiza_han_model.h5',monitor='val_acc',verbose=1,save_best_only=True)

model_history = model.fit(train_x, train_y, batch_size=256, epochs=15, validation_split=0.2,callbacks=[chkpt])

# Saving the model so that it can be loaded easily again
model.save_weights('expertiza_han_model_weights.h5')

# Save the model architecture
with open('expertiza_han_model_architecture.json', 'w') as f:
    f.write(model.to_json())

# Persisting model weights
# !cp expertiza_han_model_weights.h5 gdrive/My\ Drive/AI-In-Peer-Assessment/model/
# Persisting model architecture
# !cp expertiza_han_model_architecture.json gdrive/My\ Drive/AI-In-Peer-Assessment/model/