import numpy as np
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split

# Use Tokenizer to remove punctuations and non-word characters and tokenize the text
import os
os.environ['KERAS_BACKEND']='tensorflow' # Or TenserFlow
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.initializers import glorot_normal, orthogonal
from keras import initializers, regularizers, constraints

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

MAX_SEQUENCE_LENGTH = 124
MAX_NB_WORDS = 80 # This specifies how many top tokens in each review to be stored. Wrongly interpreted as total number of words(token) together in whole dataset
EMBEDDING_DIM = 100

# !cp ./gdrive/My\ Drive/Colab\ Notebooks/data/glove100d.zip .
# !unzip glove100d.zip

# embeddings_index = {}
# f = open('glove.6B.100d.txt', encoding="utf8")
# for line in f:
#     values = line.split()
#     word = ''.join(values[:-300])
#     coefs = np.asarray(values[-300:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()

embeddings_index = {}
for i, line in enumerate(open('glove.6B.100d.txt')):
  values = line.split() # 0 th index will be the word and rest will the embedding vector (size 100 as we have used Glove.6B.100D embedding file) 
  embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
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
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

class Capsule(Layer):

    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),   # noqa
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),   # noqa
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))    # noqa
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]  # noqa

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]  # noqa
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]    # noqa
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))    # noqa
            if i < self.routings - 1:
                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class DropConnect(wrappers.Wrapper):

    def __init__(self, layer, prob, **kwargs):
        self.prob = prob
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        if 0. < self.prob < 1.:
            self.layer.kernel = K.in_train_phase(
                K.dropout(self.layer.kernel, self.prob),
                self.layer.kernel)
            self.layer.bias = K.in_train_phase(
                K.dropout(self.layer.bias, self.prob),
                self.layer.bias)
        return self.layer.call(x)  

# Defining the Convolutional NN model

def Model1(embedding_matrix=None):
  embedding_layer = Embedding(len(word_index) + 1,
                              EMBEDDING_DIM,weights=embedding_matrix,
                              input_length=MAX_SEQUENCE_LENGTH,trainable=True)
  sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
  embedded_sequences = embedding_layer(sequence_input)
  net = SpatialDropout1D(0.2)(embedded_sequences)
  net = Bidirectional(
        layer=GRU(EMBEDDING_DIM, return_sequences=True,
                        kernel_initializer=glorot_normal(seed=1029),
                        recurrent_initializer=orthogonal(gain=1.0, seed=1029)),
        name='bidirectional_gru')(net)
  # net = Bidirectional(
  #         layer=LSTM(EMBEDDING_DIM, return_sequences=True,
  #                         kernel_initializer=glorot_normal(seed=1029),
  #                         recurrent_initializer=orthogonal(gain=1.0, seed=1029)),
  #         name='bidirectional_lstm')(net)
  #net = BatchNormalization()(net)

  capsul = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(net) # noqa
  capsul = Flatten()(capsul)
  capsul = DropConnect(Dense(8, activation="relu"), prob=0.01)(capsul)
  atten = Attention(step_dim=MAX_SEQUENCE_LENGTH, name='attention')(net)
  atten = DropConnect(Dense(4, activation="relu"), prob=0.2)(atten)
  net = Concatenate(axis=-1)([capsul, atten])
  # net = GlobalAveragePooling1D()(net)
  # output = Dense(units=1, activation='sigmoid', name='output')(net)
  # net = GlobalAveragePooling1D()(net)
  net = Dense(100, activation='relu')(net)
  # net = Dropout(0.3)(net)
  output = Dense(2, activation='softmax')(net)
  model2 = Model(inputs = sequence_input, outputs = output)
  model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
  model2.summary()
  # Keeping a checkpoint to store only the model which gives best output validation accuracy
  chkpt2=ModelCheckpoint('expertiza_nn_model2.h5',monitor='val_acc',verbose=1,save_best_only=True)
  return (model2, chkpt2)

# Defining the Convolutional NN model
def Model2(embedding_matrix=None):

  # wt = [embedding_matrix] if embedding_matrix != None else None
  embedding_layer = Embedding(len(word_index) + 1,
                              EMBEDDING_DIM,weights=embedding_matrix,
                              input_length=MAX_SEQUENCE_LENGTH,trainable=True)
  sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
  embedded_sequences = embedding_layer(sequence_input)
  net = SpatialDropout1D(0.2)(embedded_sequences)
  net = Conv1D(100, 3, padding='same', activation='relu')(net)
  net = BatchNormalization()(net)
  # net=GRU(EMBEDDING_DIM, return_sequences=True,
  #                         kernel_initializer=glorot_normal(seed=1029),
  #                         recurrent_initializer=orthogonal(gain=1.0, seed=1029))(net)
  net = Bidirectional(
          layer=GRU(100, return_sequences=True,
                          kernel_initializer=glorot_normal(seed=1029),
                          recurrent_initializer=orthogonal(gain=1.0, seed=1029)),
          name='bidirectional_gru')(net)
  # net = Bidirectional(
  #         layer=LSTM(EMBEDDING_DIM, return_sequences=True,
  #                         kernel_initializer=glorot_normal(seed=1029),
  #                         recurrent_initializer=orthogonal(gain=1.0, seed=1029)),
  #         name='bidirectional_lstm')(net)
  #net = BatchNormalization()(net)
  capsul = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(net) # noqa
  capsul = Flatten()(capsul)
  capsul = DropConnect(Dense(32, activation="relu"), prob=0.2)(capsul)
  atten = Attention(step_dim=MAX_SEQUENCE_LENGTH, name='attention')(net)
  atten = DropConnect(Dense(16, activation="relu"), prob=0.2)(atten)
  net = Concatenate(axis=-1)([capsul, atten])
  # net = GlobalAveragePooling1D()(net)
  # output = Dense(units=1, activation='sigmoid', name='output')(net)
  # net = GlobalAveragePooling1D()(net)
  net = Dense(100, activation='relu')(net)
  # net = Dropout(0.3)(net)
  output = Dense(2, activation='softmax')(net)
  model4 = Model(inputs = sequence_input, outputs = output)
  model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
  model4.summary()
  chkpt4=ModelCheckpoint('expertiza_nn_model3.h5',monitor='val_acc',verbose=1,save_best_only=True)
  return (model4, chkpt4)

def Model3(embedding_matrix=None):
  # wt = None if embedding_matrix == None else [embedding_matrix]
  embedding_layer = Embedding(len(word_index) + 1,
                              EMBEDDING_DIM,weights=embedding_matrix,
                              input_length=MAX_SEQUENCE_LENGTH,trainable=True)
  sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
  embedded_sequences = embedding_layer(sequence_input)

  net = SpatialDropout1D(0.2)(embedded_sequences)
  net = Conv1D(100, 3, padding='same', activation='relu')(net)
  net = BatchNormalization()(net)

  net=GRU(EMBEDDING_DIM, return_sequences=True,
                  kernel_initializer=glorot_normal(seed=1029),
                  recurrent_initializer=orthogonal(gain=1.0, seed=1029))(net)
  capsul = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(net) # noqa
  capsul = Flatten()(capsul)
  capsul = DropConnect(Dense(32, activation="relu"), prob=0.2)(capsul)
  atten = Attention(step_dim=MAX_SEQUENCE_LENGTH, name='attention')(net)
  atten = DropConnect(Dense(16, activation="relu"), prob=0.2)(atten)
  net = Concatenate(axis=-1)([capsul, atten])
  # net = GlobalAveragePooling1D()(net)
  # output = Dense(units=1, activation='sigmoid', name='output')(net)
  # net = GlobalAveragePooling1D()(net)
  net = Dense(100, activation='relu')(net)
  # net = Dropout(0.3)(net)
  output = Dense(2, activation='softmax')(net)
  model3 = Model(inputs = sequence_input, outputs = output)
  model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
  model3.summary()
  # Keeping a checkpoint to store only the model which gives best output validation accuracy
  chkpt3=ModelCheckpoint('expertiza_nn_model3.h5',monitor='val_acc',verbose=1,save_best_only=True)
  return (model3, chkpt3)

#  !pip install --upgrade featuretools
# pip install 'featuretools[nlp_primitives]'

from sklearn import preprocessing
import featuretools as ft
from featuretools.primitives import *
from nlp_primitives import (
    DiversityScore,
    LSA,
    MeanCharactersPerWord,
    PartOfSpeechCount,
    PolarityScore, 
    PunctuationCount,
    StopwordCount,
    TitleWordCount,
    UniversalSentenceEncoder,
    UpperCaseCount)

trans = [DiversityScore, LSA, MeanCharactersPerWord, 
         PartOfSpeechCount, PolarityScore, PunctuationCount, 
         StopwordCount, TitleWordCount,      
         UpperCaseCount]

stratified_file = "./gdrive/My Drive/Colab Notebooks/data/problems_expertiza_randomized_datasets_fullsplit/problems_expertiza_randomized_fullsplit1.csv"
df = pd.read_csv(stratified_file, engine = 'python')

def get_synthetic_data(df):
  es = ft.EntitySet('Reviews')
  es.entity_from_dataframe(dataframe=pd.concat([pd.Series(df.index, name='id'),df['REVIEW']], axis=1),
                          entity_id='review',
                          index='id')
  fm, features = ft.dfs(entityset=es, 
                        target_entity='review',
                        trans_primitives= trans)
  # df = df.drop(columns=['id'])

  x = fm.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  fm = pd.DataFrame(x_scaled)
  return fm, features

fm, featuers = get_synthetic_data(df)

df1 = pd.concat([df,fm])

import copy
f1_scores = []
accAvg = 0
for i in range(1, 31):
  stratified_file = "./gdrive/My Drive/Colab Notebooks/data/problems_expertiza_randomized_datasets_fullsplit/problems_expertiza_randomized_fullsplit" + str(i) + ".csv"
  df = pd.read_csv(stratified_file, engine = 'python');
  print("getting synthetic data")
  fm, features = get_synthetic_data(df)
  print("generated synthetic data")
  # df = pd.concat([df,fm])
  # X_columns = copy.copy(df.columns).remove('TAG')
  X_train = df["REVIEW"][:8947]
  X_val = df["REVIEW"][8947:10053]
  X_test = df["REVIEW"][10053:]
  Y_train = df["TAG"][:8947]
  Y_val = df["TAG"][8947:10053]
  Y_test = df["TAG"][10053:]
  tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
  tokenizer.fit_on_texts(X_train)
  train_sequences = tokenizer.texts_to_sequences(X_train)
  word_index = tokenizer.word_index
  print('Number of Unique Tokens',len(word_index))
  tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
  tokenizer.fit_on_texts(X_val)
  validation_sequences = tokenizer.texts_to_sequences(X_val)
  tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
  tokenizer.fit_on_texts(X_test)
  test_sequences = tokenizer.texts_to_sequences(X_test)
  word_index_2 = tokenizer.word_index
  print('Number of Unique Tokens',len(word_index_2)) # Total 996497 unique words 
  train_sequences_padded = pad_sequences(train_sequences, maxlen=100)
  validation_sequences_padded = pad_sequences(validation_sequences, maxlen=100)
  test_sequences_padded = pad_sequences(test_sequences, maxlen=100)

  from sklearn.utils import shuffle
  x_train, y_train = shuffle(train_sequences_padded, Y_train)
  x_val, y_val = shuffle(validation_sequences_padded, Y_val)
  x_test, y_test = shuffle(test_sequences_padded, Y_test)


  # concat synthetic data
  print("synthetic shapes")
  X_columns = list(fm.columns)
  print(pd.DataFrame(x_val).shape)
  print(fm.shape)
  x_train = pd.concat([pd.DataFrame(x_train),fm[:8947]],axis=1)
  x_val = pd.concat([pd.DataFrame(x_val),fm[8947:10053].reset_index(drop=True)],axis=1)
  x_test = pd.concat([pd.DataFrame(x_test),fm[10053:].reset_index(drop=True)],axis=1)

  print(x_val.shape)
  print("synthetic shapes ====")

  # Continue with your classifications approaches below
  x_train = np.array(x_train[:])
  train_labels = [[1,0] if x == 1 else [0,1] for x in y_train[:]] 
  y_train = np.array(train_labels[:])
  len(x_train),len(y_train)

  x_val = np.array(x_val[:])
  val_labels = [[1,0] if x == 1 else [0,1] for x in y_val[:]] 
  y_val = np.array(val_labels[:])
  len(x_val),len(y_val)

  x_test = np.array(x_test[:])
  test_labels = [[1,0] if x == 1 else [0,1] for x in y_test[:]] 
  y_test = np.array(test_labels[:])
  len(x_test),len(y_test)

  embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM)) # 300 since embedding_dimesion is 300, +1 because index 0 is reserved in word_index
  for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector

  model, chkpt = Model1([embedding_matrix])
  print("======================")
  print(x_train.shape)
  print(y_train.shape)
  print(x_val.shape)
  print(y_val.shape)
  print("======================")
  model_history = model.fit(x_train, y_train, batch_size=512, epochs=5, validation_data=(x_val, y_val),callbacks=[chkpt])
  
  y_pred = model.predict(x_test)
  y_test =  [1 if yt[0]==1 else -1 for yt in y_test]
  y_pred = [1 if yt[0]>yt[1] else -1 for yt in y_pred]
  print(y_test)
  print(y_pred)

  print(confusion_matrix(y_test, y_pred))
  accuracy = accuracy_score(y_test, y_pred)
  precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred,average="macro")
  f1_scores.append(f1)
  # loss, accuracy = model.evaluate(x_test, y_test, verbose = 0)
  print(accuracy)
  print(f1)
  accAvg += accuracy

print("avg accuracy "+ str(accAvg/30))
print("f1 " + str(f1_scores))

pd.concat([pd.DataFrame(x_val), fm[8947:10053].reset_index(drop=True)], axis=1).shape

# fm[8947:10053].reset_index(drop=True)

# pd.DataFrame(x_val).shape


