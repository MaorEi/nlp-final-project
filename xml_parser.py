import xml.etree.ElementTree as ETree
import pandas as pd


def getReviewsAndOpinionsFromXML(xmlFilePath):
    ''' parses the given XML file and returns reviews and opinions from it '''

    xmlTree = ETree.parse(xmlFilePath)
    treeRoot = xmlTree.getroot()
    reviews = treeRoot.findall('Review')
    extractedTokens = []
    extractedOpinions = []

    for review in reviews:
        extractedToken = ''
        extractedOpinion = []
        sentences = review.findall('./sentences/sentence')

        for sentence in sentences:
            extractedToken += ' ' + sentence.find('text').text
            opinions = sentence.findall('./Opinions/Opinion')

            for opinion in opinions:
                extractedOpinion.append({opinion.get('category') : opinion.get('polarity')})

        extractedTokens.append(extractedToken)
        extractedOpinions.append(extractedOpinion)

    return extractedTokens, extractedOpinions


def getReviewsAndOpinionsFromXMLToCSV(xmlFilePath, csvFilePath):
    ''' extracts reviews and opinions from XML and writes into CSV file '''

    xmlTree = ETree.parse(xmlFilePath)
    treeRoot = xmlTree.getroot()
    reviews = treeRoot.findall('Review')

    reviewsArray = []
    aspectsArray = []
    sentimentsArray = []

    for review in reviews:
        sentences = review.findall('./sentences/sentence')
        for sentence in sentences:
            opinions = sentence.findall('./Opinions/Opinion')
            for opinion in opinions:
                reviewsArray.append(sentence.find('text').text)
                aspectsArray.append(opinion.get('category'))
                sentimentsArray.append(opinion.get('polarity'))

    data = {'review': reviewsArray, 
            'aspect': aspectsArray, 
            'sentiment': sentimentsArray}

    data = pd.DataFrame(data, columns=data.keys())
    data.to_csv(csvFilePath, index=None, header=True)


def convertXMLtoCSV(xmlFilePath, csvFilePath):
    ''' extracts reviews and opinions from XML and writes opinions (aspects and sentiments) into CSV file as one column (AMBIENCE#GENERAL:positive) '''
    
    xmlTree = ETree.parse(xmlFilePath)
    treeRoot = xmlTree.getroot()
    reviews = treeRoot.findall('Review')

    reviewsArray = []
    aspectsArray = []
    categoryArray = []
    sentimentsArray = []

    for review in reviews:
        sentences = review.findall('./sentences/sentence')
        for sentence in sentences:
            opinions = sentence.findall('./Opinions/Opinion')
            for opinion in opinions:
                reviewsArray.append(sentence.find('text').text)
                aspectsArray.append(opinion.get('target'))
                categoryArray.append(opinion.get('category'))
                sentimentsArray.append(opinion.get('polarity'))

    data = {'review': reviewsArray, 
            'aspect': aspectsArray,
            'category': [category + ':' + sentiment for category, sentiment in zip(categoryArray, sentimentsArray)]}

    data = pd.DataFrame(data, columns=data.keys())
    data.to_csv(csvFilePath, index=None, header=True)