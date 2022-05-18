from flask import Flask, request
import gensim
from utils import preprocess
import os
from waitress import serve
app = Flask(__name__)


@app.route("/classify", methods=['GET', 'POST'])
def classify():
    score_list={}
    path = os.path.curdir + '/LdaModel/lda.model'
    dict_path = os.path.curdir + '/LdaModel' + '/dict_file'
    try:
        if "user_query" in request.json:
            user_query = request.json['user_query']

        processed_docs = preprocess(user_query)
        dictionary=gensim.corpora.Dictionary.load(dict_path)
        lda_model = gensim.models.LdaMulticore.load(fname=path, mmap='r')
        bow_vector = dictionary.doc2bow(processed_docs)

        for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
            score_list["Score" + str(index)] = ("Score: {} " " Topic: {}".format(score, lda_model.print_topic(index, 5)))
        return score_list

    except Exception as e:
        return "Bad Request/ Internal Error" + str(e)


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8991, threads=10)
