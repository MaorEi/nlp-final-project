from pathlib import Path
from xml_parser import getReviewsAndOpinionsFromXML, getReviewsAndOpinionsFromXMLToCSV

### XML and CSV files path ###
testingXMLPath = str(Path().absolute()) + '/Data/EN_REST_SB1_TEST.xml.gold'
testingCSVPath = str(Path().absolute()) + '/Data/test.csv'
testingFoursquareCSVPath = str(Path().absolute()) + '/Data/testFoursquare.csv'
foursquareTestingXMLPath = str(Path().absolute()) + '/Data/foursquare_gold.xml'
outputOpinionsFile = str(Path().absolute()) + '/Data/opinions.csv'
foursquareOutputOpinionsFile = str(Path().absolute()) + '/Data/foursquare_opinions.csv'
#######################

def drive(parser_class, first_output_opinions_file = outputOpinionsFile, second_output_opinions_file = foursquareOutputOpinionsFile):
    ''' drives the parser: first for training, 
    second for testing reviews from the same SemEval dataset
    and third for testing reviews from FourSquare as different dataset '''

    parser = parser_class()
    parser.train() 
    
    getReviewsAndOpinionsFromXMLToCSV(testingXMLPath, testingCSVPath) #writes reviews and opinions into user friendly csv file. 
    reviews, opinions = getReviewsAndOpinionsFromXML(testingXMLPath)
    parser.test(reviews, opinions, first_output_opinions_file) #test reviews from SemEval (the same dataset as training).

    getReviewsAndOpinionsFromXMLToCSV(foursquareTestingXMLPath, testingFoursquareCSVPath) #writes reviews and opinions into user friendly csv file.
    reviews, opinions = getReviewsAndOpinionsFromXML(foursquareTestingXMLPath)
    parser.test(reviews, opinions, second_output_opinions_file) #test reviews from FourSquare (different dataset).
