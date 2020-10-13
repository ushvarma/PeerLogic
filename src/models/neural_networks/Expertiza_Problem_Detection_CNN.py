import numpy as np
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        

os.environ['KERAS_BACKEND']='tensorflow' # Or TenserFlow       

MAX_SEQUENCE_LENGTH = 100
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

#!cp gdrive/My\ Drive/glove100d.zip .
#!unzip glove100d.zip

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

x_train, y_train = shuffle(train_sequences_padded, train_labels)

x_train = np.array(x_train[:])
train_labels = [[1,0] if x == 1 else [0,1] for x in y_train[:]] 
y_train = np.array(train_labels[:])

# Defining the Convolutional NN model
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

net = Dropout(0.3)(embedded_sequences)
net = Conv1D(50, 3, padding='same', activation='relu')(net)
net = BatchNormalization()(net)
net = GlobalAveragePooling1D()(net)
net = Dense(100, activation='relu')(net)
net = Dropout(0.5)(net)
output = Dense(2, activation='softmax')(net)

model = Model(inputs = sequence_input, outputs = output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# Keeping a checkpoint to store only the model which gives best output validation accuracy
chkpt=ModelCheckpoint('expertiza_cnn_model.h5',monitor='val_acc',verbose=1,save_best_only=True)
model_history = model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.2,callbacks=[chkpt])

# Saving the model so that it can be loaded easily again
model.save_weights('expertiza_cnn_model_weights.h5')

# Save the model architecture
with open('expertiza_cnn_model_architecture.json', 'w') as f:
    f.write(model.to_json())

# Persisting model weights
# !cp expertiza_cnn_model_weights.h5 gdrive/My\ Drive/AI-In-Peer-Assessment/model/

# Persisting model architecture
# !cp expertiza_cnn_model_architecture.json gdrive/My\ Drive/AI-In-Peer-Assessment/model/

# !cp gdrive/My\ Drive/AI-In-Peer-Assessment/model/expertiza_cnn_model_architecture.json .
# !cp gdrive/My\ Drive/AI-In-Peer-Assessment/model/expertiza_cnn_model_weights.h5 .

# Loading model and weights again from drive
from keras.models import model_from_json

# Model reconstruction from JSON file
with open('expertiza_cnn_model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('expertiza_cnn_model_weights.h5')

allclassifierF1Score = []
allclassifierF1Score.append(classifierF1ScoreCNN)
allclassifierF1Score.append(classifierF1ScoreRNN)
allclassifierF1Score.append(classifierF1ScoreCNN_plus_RNN)
allclassifierF1Score.append(classifierF1ScoreHAN)

allclassifierF1Score
classifier_names = ["CNN","LSTM","CNN_Plus_LSTM","HAN"]

stats_df = pd.DataFrame(allclassifierF1Score)
stats_df = stats_df.transpose()
stats_df.columns = classifier_names

plt.clf()
fig1, ax1 = plt.subplots()
ax1.set_title("NN Classifier F1-scores")
sns.boxplot(x="variable", y="value", data=pd.melt(stats_df))
plt.xlabel("Classifiers")
plt.ylabel("F1-scores")
plt.show()