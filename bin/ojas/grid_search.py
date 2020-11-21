import pandas as pd
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras import layers
from keras import optimizers

import matplotlib.pyplot as plt
plt.style.use('ggplot')


#file path
filepath = "suggestions_expertiza_fall2018.csv"

df = pd.read_csv(filepath, encoding = "ISO-8859-1")

df.drop_duplicates(keep=False, inplace=True)
print(df.keys())

sentences = df['Comments'].values


df['value'] = df['Suggest_Solutions'].replace([-1], 0)
values = df["value"].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
                    sentences, values, test_size=0.15, random_state=1000)


vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy:", score)


#define grid search parameters
batch_sizes = [8, 16, 32, 64, 128]
learn_rates = [0.0001, 0.001, 0.01, 0.1]


input_dim = X_train.shape[1]  # Number of features

def plot_history(history, filename):
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
    plt.savefig(filename+".png")
    plt.show()


for learn_rate in learn_rates:
    for batch_size in batch_sizes:

        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        adam = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        model.summary()

        history = model.fit(X_train, y_train,
                            epochs=20,
                            verbose=True,
                            validation_data=(X_test, y_test),
                            batch_size=batch_size)
        with open("output.txt","a") as file:
            file.write("Batch Size: {0} Learning Rate: {1} \n".format(batch_size, learn_rate))
            loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
            file.write("Training Accuracy: {:.4f}".format(accuracy))
            loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
            file.write("Testing Accuracy:  {:.4f}".format(accuracy))
            file.write("\n\n")
        plot_history(history, "Batch Size: {0} Learning Rate: {1} \n".format(batch_size, learn_rate))



