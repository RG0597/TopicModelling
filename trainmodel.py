from flask import Flask, request
import pandas as pd
import os
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
from utils import preprocess
np.random.seed(2018)

app = Flask(__name__)
stemmer = SnowballStemmer('english')

@app.route("/train", methods=['GET', 'POST'])
def model_train():
    path = os.path.curdir + '/LdaModel' + '/lda.model'
    dict_path = os.path.curdir + '/LdaModel' + '/dict_file'
    if request.method == 'POST':
        # uploading any file to train the model
        f = request.files['fileupload']
        data = pd.read_csv(f)
        processed_docs = data['headline_text'].map(preprocess)
        dictionary = gensim.corpora.Dictionary(processed_docs)
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        #saving dictionary corpora
        dictionary.save(dict_path)
        # loading dictionary corpora
        dictionary=gensim.corpora.Dictionary.load(dict_path)
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
        lda_model.save(fname=path)

    return "model trained successfully"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8990)