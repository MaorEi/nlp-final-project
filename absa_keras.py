import spacy
NLP = spacy.load('en')

import string
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import tensorflow
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from xml_parser import convertXMLtoCSV

class Model():

    ### global variables ###
    trainingXMLPath = str(Path().absolute()) + '/Data/ABSA16_Restaurants_Train_SB1_v2.xml'
    trainingCSVPath = str(Path().absolute()) + '/Data/train.csv'

    aspectsTokenizer = None
    aspectsModel = None
    aspectsLabelsEncoder = None
    
    categories = []
    MAX_VOCAB_SIZE = 6000
    #########################

    def __init__(self):
        ''' Converts XML data to CSV table first for easy loading '''

        convertXMLtoCSV(self.trainingXMLPath, self.trainingCSVPath)


    def train(self, training_xml_file=trainingXMLPath):
        print('\nkeras train\n')
        self.trainABSAmodel()

    def trainABSAmodel(self):
        ''' Creates, adds layers and trains the keras model  '''
        trainData = pd.read_csv(self.trainingCSVPath).astype(str)
        self.categories = sorted(trainData.category.unique())
        
        ABSAmodel = Sequential()
        ABSAmodel.add(Dense(512, input_shape=(6000, ), activation='relu'))
        ABSAmodel.add(Dense(256, activation='relu'))
        ABSAmodel.add(Dense(128, activation='relu'))
        ABSAmodel.add(Dense(trainData.category.nunique() , activation='softmax'))
        
        ABSAmodel.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        tokenizer = Tokenizer(num_words=self.MAX_VOCAB_SIZE)
        tokenizer.fit_on_texts(trainData.review)
        tokenizedReviews = pd.DataFrame(tokenizer.texts_to_matrix(trainData.review))

        encoder = LabelEncoder()
        intCategory = encoder.fit_transform(trainData.category)
        encodedY = to_categorical(intCategory)

        ABSAmodel.fit(tokenizedReviews, encodedY, epochs=10, verbose=1)

        self.aspectsTokenizer = tokenizer
        self.aspectsLabelsEncoder = encoder
        self.aspectsModel = ABSAmodel


    def test(self, sentences):
        ''' Tests the keras model on given sentences '''
        print('\nkeras test\n')
        
        reviews = [sentence.lower() for sentence in sentences]
        aspects = []

        #POS, punctuation and stop words filtering
        for review in NLP.pipe(reviews):
            chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']

            if review.is_parsed:
                chunks.append(' '.join([token.lemma_ for token in review
                if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
            else:
                chunks.append('')

            aspects.append(' '.join(chunks))

        aspects = pd.DataFrame(self.aspectsTokenizer.texts_to_matrix(aspects))

        return self.aspectsModel.predict(aspects)
