# TopicModelling

The repo aims at achieving topic modelling through lda via sklearn library. The dataset used is abc-news dataset. 
The purpose is to perform and and get understading of an unsuprevised form of topic modelling.
Trainmodel.py consumes a files of csv format (dataset) to train and save the trained model in a sepearte Ldamodel folder.

Classifier API aims at using the trained model in Ldamodel  folder for purpose of classification of any json format document into respective topics.


steps to follow:
1.  Clone the TopicModellling repo anywhere in your system.
2.  Make a folder "LdaModel" in the directory cloned.
3.  Run trainmodel.py using python trainmodel.py
4.  For runnning classifier.py , use following command: waitress-serve --port:8991 classifier:app
