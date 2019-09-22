from spec import Spec
from pathlib import Path
from xml_parser import getReviewsAndOpinionsFromXML
from absa_keras import Model as kerasModel
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn import metrics
from scipy.sparse import hstack
from nltk import word_tokenize
from nltk import pos_tag
import nltk
import ssl
import warnings
import numpy as np
import pandas as pd

class Model(Spec):

    ### global variables ###
    trainingXMLPath = str(Path().absolute()) + '/Data/ABSA16_Restaurants_Train_SB1_v2.xml'
    commonAspects = []
    YTrain = None
    YTest = None
    trainDataFrame = None
    testDataFrame = None
    DLKerasModel = None
    positivePrediction = None
    negativePrediction = None
    neutralPrediction = None
    #########################

    def __init__(self):
        ''' checks if necessary packages are installed... '''

        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

    def train(self, training_xml_file = trainingXMLPath):
        ''' creates data frames from the reviews and opinions in the SemEval restaurants domain training dataset '''

        print('\ntrain\n')
        reviews, opinions = getReviewsAndOpinionsFromXML(training_xml_file)

        self.commonAspects = self.getCommonAspects(opinions)

        taggedReviews = self.POSTagging(reviews)
        taggedReviews = self.filterTagging(taggedReviews)

        dataFrame = self.createDataFrame(taggedReviews, opinions, self.commonAspects)
        aspectDataFrame = self.createAspectDataFrame(dataFrame, self.commonAspects)
        aspectDataFrame = aspectDataFrame.reindex(sorted(aspectDataFrame.columns), axis = 1)

        self.YTrain = aspectDataFrame.drop('Review', 1)
        self.YTrain = np.asarray(self.YTrain, dtype = np.int64)

        self.trainDataFrame = dataFrame

    def test(self, sentences, opinions, output_opinions_file):
        ''' creates data frame from the given sentences and opinions and write the parsing result to the given file path '''

        print('\ntest\n')
        
        taggedSentences = self.POSTagging(sentences)
        taggedSentences = self.filterTagging(taggedSentences)

        dataFrame = self.createDataFrame(taggedSentences, opinions, self.commonAspects)
        aspectDataFrame = self.createAspectDataFrame(dataFrame, self.commonAspects)
        aspectDataFrame = aspectDataFrame.reindex(sorted(aspectDataFrame.columns), axis = 1)

        self.YTest = aspectDataFrame.drop('Review', 1)
        self.YTest = np.asarray(self.YTest, dtype = np.int64)

        self.testDataFrame = dataFrame

        #first round:
        print('\n-----------')
        print('first round')
        print('-----------\n')
        parsing = self.parse()

        firstRoundFilePath = Path(output_opinions_file)

        #create empty csv file
        with open(firstRoundFilePath, 'w'):
            pass

        firstRoundFilePath.rename(Path(firstRoundFilePath.parent, f'{firstRoundFilePath.stem}_first_round{firstRoundFilePath.suffix}'))
        firstRoundFilePath = f'{firstRoundFilePath.parent}/{firstRoundFilePath.stem}_first_round{firstRoundFilePath.suffix}'
        print('\nwriting the result parsing to file: {}\n'.format(firstRoundFilePath))
        self.writeParsingResultToCSV(sentences, parsing, firstRoundFilePath)

        #second round:
        print('\n------------')
        print('second round')
        print('------------\n')
        self.activateSecondRoundModel(sentences)
        parsing = self.parse()

        secondRoundFilePath = Path(output_opinions_file)

        #create empty csv file
        with open(secondRoundFilePath, 'w'):
            pass

        secondRoundFilePath.rename(Path(secondRoundFilePath.parent, f'{secondRoundFilePath.stem}_second_round{secondRoundFilePath.suffix}'))
        secondRoundFilePath = f'{secondRoundFilePath.parent}/{secondRoundFilePath.stem}_second_round{secondRoundFilePath.suffix}'
        print('\nwriting the result parsing to file: {}\n'.format(secondRoundFilePath))
        self.writeParsingResultToCSV(sentences, parsing, secondRoundFilePath)

        self.positivePrediction = None
        self.negativePrediction = None
        self.neutralPrediction = None

    def parse(self):
        ''' builds the data that is necessary to the parsing function '''

        trainAspectDict = self.createDictForAspects(self.YTrain, self.commonAspects)
        DTrain = DictVectorizer()
        XTrainTransform = DTrain.fit_transform(trainAspectDict)

        testAspectDict = self.createDictForAspects(self.YTest, self.commonAspects)
        DTest = DictVectorizer()
        XTestTransform = DTest.fit_transform(testAspectDict)

        trainPositiveDataFrame = self.createPositiveOpinionsDataFrame(self.trainDataFrame, self.commonAspects)
        testPositiveDataFrame = self.createPositiveOpinionsDataFrame(self.testDataFrame, self.commonAspects)
        
        trainNegativeDataFrame = self.createNegativeOpinionsDataFrame(self.trainDataFrame, self.commonAspects)
        testNegativeDataFrame = self.createNegativeOpinionsDataFrame(self.testDataFrame, self.commonAspects)

        trainNeutralConflictDataFrame = self.createNeutralConflictOpinionsDataFrame(self.trainDataFrame, self.commonAspects)
        testNeutralConflictDataFrame = self.createNeutralConflictOpinionsDataFrame(self.testDataFrame, self.commonAspects)

        print('positive classification:')
        positiveClassification = self.aspectsClassification(trainPositiveDataFrame, testPositiveDataFrame, XTrainTransform, XTestTransform, Sentiment.POSITIVE)

        print('negative classification:')
        negativeClassification = self.aspectsClassification(trainNegativeDataFrame, testNegativeDataFrame, XTrainTransform, XTestTransform, Sentiment.NEGATIVE)

        print('neutral conflict classification:')
        neutralConflictClassification = self.aspectsClassification(trainNeutralConflictDataFrame, testNeutralConflictDataFrame, XTrainTransform, XTestTransform, Sentiment.NEUTRAL)

        parsing = self.getParsingForClassification(positiveClassification, negativeClassification, neutralConflictClassification)

        return parsing


    def aspectsClassification(self, trainDataFrame, testDataFrame, XTrainAspectTransform, XTestAspectTransform, sentiment):
        ''' uses svm algorithm to predict aspects classification '''

        trainDataFrame = trainDataFrame.reindex(sorted(trainDataFrame.columns), axis=1)
        testDataFrame = testDataFrame.reindex(sorted(testDataFrame.columns), axis=1)

        XTrain = trainDataFrame.Review
        YTrain = trainDataFrame.drop('Review',1)
        YTrain = np.asarray(YTrain, dtype=np.int64)

        XTest = testDataFrame.Review
        YTest = testDataFrame.drop('Review',1)
        YTest = np.asarray(YTest, dtype=np.int64)

        vectorizer = CountVectorizer(stop_words='english',ngram_range=(1,2))  
        XTrainTransform = vectorizer.fit_transform(XTrain)
        XTestTransform = vectorizer.transform(XTest)

        XTrainTransform = hstack((XTrainTransform, XTrainAspectTransform))
        XTestTransform = hstack((XTestTransform, XTestAspectTransform))
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            SVC = OneVsRestClassifier(svm.SVC(kernel='linear', C=1.0)).fit(XTrainTransform, YTrain)
            SVCYPrediction = SVC.predict(XTestTransform)

            if sentiment == Sentiment.POSITIVE:
                if self.positivePrediction is None:
                    self.positivePrediction = SVCYPrediction
                else:
                    SVCYPrediction = self.positivePrediction

            elif sentiment == Sentiment.NEGATIVE:
                if self.negativePrediction is None:
                    self.negativePrediction = SVCYPrediction
                else:
                    SVCYPrediction = self.negativePrediction

            else: #sentiment == Sentiment.NEUTRAL:
                if self.neutralPrediction is None:
                    self.neutralPrediction = SVCYPrediction
                else:
                    SVCYPrediction = self.neutralPrediction

            print("Accuracy:")
            print(metrics.accuracy_score(YTest, SVCYPrediction))
                
            print("\nAverage precision:")
            print(metrics.precision_score(YTest, SVCYPrediction, average='micro'))

            print("\nAverage recall:")
            print(metrics.recall_score(YTest, SVCYPrediction, average='micro'))
                
            print("\nAverage f1:")
            print(metrics.f1_score(YTest,SVCYPrediction,average='micro'))

            print("\nClassification report:")
            print(metrics.classification_report(YTest, SVCYPrediction, target_names=self.commonAspects))

        return SVCYPrediction


    def POSTagging(self, sentence):
        ''' return POS tagging for each word in the given sentence '''

        taggedTokens = []

        for word in sentence:
            taggedTokens.append(pos_tag(word_tokenize(word)))

        return taggedTokens

    def filterTagging(self, taggedSentence):
        ''' returns only relevant words according to POS '''

        filteredSentence = []

        for words in taggedSentence:
            filtered = []
            for word, tag in words:
                if tag in ['NN','NNS','NNP','NNPS','RB','RBR','RBS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ']:
                    filtered.append(word)

            filteredSentence.append(' '.join(filtered))

        return filteredSentence
    
    def getCommonAspects(self, opinions):
        ''' returns the most common aspects (up to 20) '''

        currentOpinion = []

        for opinion in opinions:
            for opinionDict in opinion:
                for aspect in opinionDict:
                    currentOpinion.append(aspect)

        return sorted([key for key, _ in nltk.FreqDist(currentOpinion).most_common(20)])

    def createDataFrame(self, reviews, opinions, commonAspects):
        ''' creates pandas data frame from the given reviews and opinions '''

        data = {'Review' : reviews}
        dataFrame = pd.DataFrame(data)

        for opinion in opinions:
            for opinionDict in opinion:
                for aspect in opinionDict:
                    if aspect in commonAspects:
                        dataFrame.loc[opinions.index(opinion), aspect] = opinionDict[aspect]

        return dataFrame

    def createAspectDataFrame(self, dataFrame, commonAspects):
        ''' replaces all polarity words to 1, 0 otherwise '''

        for aspect in commonAspects:
            dataFrame = dataFrame.replace({aspect : {'positive' : 1, 'negative' : 1, 'neutral' : 1, 'conflict' : 1}})

        return dataFrame.fillna(0)

    def createPositiveOpinionsDataFrame(self, dataFrame, commonAspects):
        ''' replaces positive polarity to 1, 0 otherwise '''

        for aspect in commonAspects:
            dataFrame = dataFrame.replace({aspect : {'positive' : 1, 'negative' : 0, 'neutral' : 0, 'conflict' : 0}})

        return dataFrame.fillna(0)

    def createNegativeOpinionsDataFrame(self, dataFrame, commonAspects):
        ''' replaces negative polarity to 1, 0 otherwise '''

        for aspect in commonAspects:
            dataFrame = dataFrame.replace({aspect : {'positive' : 0, 'negative' : 1, 'neutral' : 0, 'conflict' : 0}})

        return dataFrame.fillna(0)

    def createNeutralConflictOpinionsDataFrame(self, dataFrame, commonAspects):
        ''' replaces neutral conflict polarity to 1, 0 otherwise '''

        for aspect in commonAspects:
            dataFrame = dataFrame.replace({aspect : {'positive' : 0, 'negative' : 0, 'neutral' : 1, 'conflict' : 1}})

        return dataFrame.fillna(0)

    def createDictForAspects(self, YVector, commonAspects):
        ''' checks if the aspects appear in the vector and creates aspects dictionary '''

        index = []
        aspectsDict = []
        sortedAspects = sorted(commonAspects)

        for y in YVector:
            index.append([i for i, j in enumerate(y) if j == 1])

        for indexList in index:
            indexDict = {}
            for aspect in sortedAspects:
                if sortedAspects.index(aspect) in indexList:
                    indexDict[aspect] = 5
                else:
                    indexDict[aspect] = 0

            aspectsDict.append(indexDict)

        return aspectsDict


    def getParsingForClassification(self, positiveClassification, negativeClassification, neutralConflictClassification):
        ''' translates classification to actual parsing in polarity words: "positive", "negative", "neutral conflict" '''

        positiveIndexes = defaultdict(list)
        negativeIndexes = defaultdict(list)
        neutralConflictIndexes = defaultdict(list)
        parsing = defaultdict(list)

        for (j, row) in enumerate(positiveClassification.tolist()):
            for (i, value) in enumerate(row):
                if value:
                    positiveIndexes[j].append(i)
                    
            
        for (j, row) in enumerate(negativeClassification.tolist()):
            for (i, value) in enumerate(row):
                if value:
                    negativeIndexes[j].append(i)

        for (j, row) in enumerate(neutralConflictClassification.tolist()):
            for (i, value) in enumerate(row):
                if value:
                    neutralConflictIndexes[j].append(i)

        for j, row in positiveIndexes.items():
            for i in row:
                parsing[j].append(sorted(self.commonAspects)[i] + ' - positive')

        for j, row in negativeIndexes.items():
            for i in row:
                parsing[j].append(sorted(self.commonAspects)[i] + ' - negative')

        for j, row in neutralConflictIndexes.items():
            for i in row:
                parsing[j].append(sorted(self.commonAspects)[i] + ' - neutral conflict')

        return parsing


    def activateSecondRoundModel(self, reviews):
        ''' Trains and tests the deep learning keras model '''

        positivePredictions = np.zeros(self.positivePrediction.shape, dtype=np.int64)
        negativePredictions = np.zeros(self.negativePrediction.shape, dtype=np.int64)
        neutralPredictions = np.zeros(self.neutralPrediction.shape, dtype=np.int64)

        keras = None 
        if self.DLKerasModel is None:
            keras = kerasModel()
            keras.train()
        else:
            keras = self.DLKerasModel

        kerasPrediction = keras.test(reviews)
        
        for j, predictions in enumerate(kerasPrediction):
            for i, prediction in enumerate(predictions):
                if prediction >= 0.2:
                    category = keras.categories[i].split(':')
                    sentiment = category.pop()
                    category = category.pop()
                    idx = self.commonAspects.index(category)

                    if sentiment == Sentiment.POSITIVE:
                        positivePredictions[j][idx] = 1
                    elif sentiment == Sentiment.NEGATIVE:
                        negativePredictions[j][idx] = 1
                    else: # sentiment == Sentiment.NEUTRAL
                        neutralPredictions[j][idx] = 1

        self.positivePrediction |= positivePredictions
        self.negativePrediction |= negativePredictions
        self.neutralPrediction |= neutralPredictions
        self.DLKerasModel = keras


    def writeParsingResultToCSV(self, sentences, parsing, csvFilePath):
        ''' Writes the parsing result to CSV file '''

        paddingSentences = sentences.copy()
        paddingCounter = 1
        aspectsArray = []
        sentimentsArray = []

        for idx, _ in enumerate(sentences):
            for _ in range(len(parsing[idx])-1):
                paddingSentences.insert(idx+paddingCounter, '\"---\"')
                paddingCounter += 1

            if len(parsing[idx]):
                for currentParsing in parsing[idx]:
                    aspectAndSentiment = currentParsing.split(' - ')
                    aspectsArray.append(aspectAndSentiment[0])
                    sentimentsArray.append(aspectAndSentiment[1])
            else:
                aspectsArray.append('n/a')
                sentimentsArray.append('n/a')

        data = {'review': paddingSentences, 
                'aspect': aspectsArray, 
                'sentiment': sentimentsArray}

        data = pd.DataFrame(data, columns=data.keys())
        data.to_csv(csvFilePath, index=None, header=True)


from enum import Enum
class Sentiment(Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'
