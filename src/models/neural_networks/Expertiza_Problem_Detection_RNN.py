import numpy as np
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
import os
os.environ['KERAS_BACKEND']='tensorflow' # Or TenserFlow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle

reviews = pd.read_csv("problems_expertiza_merged_gabe.csv") 
data = reviews.filter(["REVIEW","TAG"])
data = data.loc[data.REVIEW.apply(lambda x: not isinstance(x, (float, int)))]

train_set, test_set = train_test_split(data, test_size=0.05, random_state=42)
train_reviews = list(train_set["REVIEW"])
train_labels = list(train_set["TAG"])

for i in range(len(train_reviews)):
  train_reviews[i] = re.sub('\d',' ',train_reviews[i]) # Replacing digits by space
  train_reviews[i] = re.sub(r'\s+[a-z][\s$]', ' ',train_reviews[i]) # Removing single characters and spaces alongside
  train_reviews[i] = re.sub(r'\s+', ' ',train_reviews[i]) # Replacing more than one space with a single space

for i in range(len(train_reviews)):
    if 'www.' in train_reviews[i] or 'http:' in train_reviews[i] or 'https:' in train_reviews[i] or '.com' in train_reviews[i]:
        train_reviews[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_reviews[i])
        
        
# train_reviews[1:5]

MAX_SEQUENCE_LENGTH = 80
MAX_NB_WORDS = 80 # This specifies how many top tokens in each review to be stored. Wrongly interpreted as total number of words(token) together in whole dataset
EMBEDDING_DIM = 100

# Takes 5 minutes to run on entire training dataset
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_reviews)
train_sequences = tokenizer.texts_to_sequences(train_reviews)

word_index = tokenizer.word_index
print('Number of Unique Tokens',len(word_index)) # Total 996497 unique words 

#Padding
train_sequences_padded = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)

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


x_train, y_train = shuffle(train_sequences_padded, train_labels)

x_train = np.array(x_train[:])
train_labels = [[1,0] if x == 1 else [0,1] for x in y_train[:]] 
y_train = np.array(train_labels[:])
# len(x_train),len(y_train)

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

net = Dropout(0.3)(embedded_sequences)
net = Bidirectional(LSTM(200,recurrent_dropout=0.4))(net)
net = Dropout(0.3)(net)
output = Dense(2, activation = 'softmax')(net)
model = Model(inputs = sequence_input, outputs = output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# Keeping a checkpoint to store only the model which gives best output validation accuracy
chkpt=ModelCheckpoint('expertiza_rnn_model.h5',monitor='val_acc',verbose=1,save_best_only=True)
model_history = model.fit(x_train, y_train, batch_size=256, epochs=10, validation_split=0.2,callbacks=[chkpt])

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_input)

net1 = Dropout(0.3)(embedded_sequences_1)
net1 = Conv1D(50, 3, padding='same', activation='relu')(net1)
net1 = AveragePooling1D(pool_size=4)(net1)
net1 = LSTM(100, recurrent_dropout=0.3)(net1)
net1 = Dropout(0.2)(net1)
output1 = Dense(2, activation='softmax')(net1)

model5 = Model(inputs = sequence_input, outputs = output1)
model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model5.summary()

# Keeping a checkpoint to store only the model which gives best output validation accuracy
chkpt=ModelCheckpoint('expertiza_cnn_rnn_model.h5',monitor='val_acc',verbose=1,save_best_only=True)
model_history1 = model5.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.1,callbacks=[chkpt])

# Saving the model so that it can be loaded easily again
model.save_weights('expertiza_rnn_model_weights.h5')

# Save the model architecture
with open('expertiza_rnn_model_architecture.json', 'w') as f:
    f.write(model.to_json())

# Persisting model weights
# !cp expertiza_rnn_model_weights.h5 gdrive/My\ Drive/AI-In-Peer-Assessment/model/
# Persisting model architecture
# !cp expertiza_rnn_model_architecture.json gdrive/My\ Drive/AI-In-Peer-Assessment/model/