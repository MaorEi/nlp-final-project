
ONLP Final Project: Aspect Based Sentiment Analysis - Domain Adaptation
(See ONLP_Final_Project.pdf)


Rotem Yagil 
ID: 026515973
Email: rotem.yagil@gmail.com

Maor Eitan
ID: 308047943
Email: maor.eitan.me@gmail.com



Developed and tested with Python 3.7.0

1. Install all required libraries using:

	pip install -r requirements.txt


2. Run the project using:

	python go.py

   For explicit Python version 3 running, use python3 go.py 


The Data folder contains XML dataset files and parsing output CSV files:

1. SemEval training dataset - ABSA16_Restaurants_Train_SB1_v2.xml

2. SemEval testing dataset - EN_REST_SB1_TEST.xml.gold

3. Foursquare testing dataset - foursquare_gold.xml

4. SemEval training dataset output CSV file (for comfort reading...) - train.csv

5. SemEval testing dataset output CSV file - test.csv

6. Foursquare testing dataset output CSV file - testFoursquare.csv

7. Parsing results output CSV file for SemEval testing first round: opinions_first_round.csv

8. Parsing results output CSV file for SemEval testing second round: opinions_second_round.csv

9. Parsing results output CSV file for Foursquare testing first round: foursquare_opinions_first_round.csv

10. Parsing results output CSV file for Foursquare testing second round: foursquare_opinions_second_round.csv