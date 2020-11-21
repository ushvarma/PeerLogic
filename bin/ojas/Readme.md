# Requirements

The following library requirements are needed. Using Conda is strongly recommended.

```
pandas
keras
numpy
matplotlib
sklearn
seaborn
```

# Data Preprocessing

The cooments contain html tags which need to be removed from the data set.

Run data_preprocessing.py which will generate two files namely suggestions_expertiza_fall2018.csv and problems_expertiza_fall2018.csv in the current directory.
These will be used for the network training and testing.


```
python3 data_preprocessing.py
```

# Models

Neural Network model

```
python3 nn.py
```

Convolutional Neural Network

```
python3 cnn.py
```
