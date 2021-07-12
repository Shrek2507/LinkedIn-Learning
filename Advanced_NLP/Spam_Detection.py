import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score
import string
import re


class SpamDetection(object):
    def __init__(self):
        self.spamData = None
        self.stopwords = list(stopwords.words("english"))
        self.punctuation = string.punctuation

    # func to read the dataset...
    def readData(self, filepath):
        print("Reading the dataset...")
        messages = pd.read_csv(filepath, encoding='latin-1')
        print("Data is read successfully.")

        # Drop the unused columns...
        messages = messages.drop(
            labels=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

        self.spamData = messages
        print("Shape Of the Dataset is : ", self.spamData.shape)

        print()
        print("Columns of the dataset are : ", self.spamData.columns)

        # Get the distribution of Spam and Ham messages...
        print("Distribution of Spam and Ham are : ")
        print(self.spamData['label'].value_counts())

    # Cleaning and processing data...
    def cleanData(self, text):
        # Perform data cleaning and transformation...
        text = "".join([word.lower()
                       for word in text if word not in self.punctuation])

        tokens = re.split('\W+', text)
        text = [word for word in tokens if word not in self.stopwords]

        return text

    # Preprocess the text...
    def preProcessText(self):
        tfidfVect = TfidfVectorizer(analyzer=self.cleanData)

        # transform the dataset...
        X_tfidf = tfidfVect.fit_transform(self.spamData['text'])
        print("Shape of the DTM : ", X_tfidf.shape)

        # get the features in a dataframe....
        X_features = pd.DataFrame(X_tfidf.toarray())

        return X_features

    # Prepare the Model...
    def buildModel(self, X_features):
        # Split the data into train and test...
        print("Splitting the data")
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, self.spamData['label'], test_size=0.2)
        print('Data is split into Train and Test')
        print()
        print('Building the Model..')
        rf = RandomForestClassifier()
        rf_model = rf.fit(X_train, y_train)
        print('Model is built.')


obj_sd = SpamDetection()
filePath = "spam.csv"
obj_sd.readData(filePath)
# obj_sd.cleanData()
X_features = obj_sd.preProcessText()
obj_sd.buildModel(X_features)
