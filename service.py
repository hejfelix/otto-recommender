from flask import Flask, g
import time
from gensim.models import KeyedVectors
import json

app = Flask(__name__)

print("Reading Word2Vec word vectors...")
wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')

print("Reading ten most popular products...")
ten_most_popular = list(json.load(open("ten_most_popular.json", "r")).keys())


@app.before_request
def before_request():
    g.start = time.time()

@app.after_request
def after_request(response):
    diff = time.time() - g.start
    if response.response:
        response.headers["X-Response-Time"] = str(diff)
    print(f"Request took {diff*1000} milliseconds")
    return response


@app.route("/api/v1/recommendations/<aid>")
def hello_world(aid):
    print(f"Received request for {aid}")
    if wv.__contains__(int(aid)):
        neighbors = wv.similar_by_vector(int(aid), topn=10)
        recommendations = [item for item, score in neighbors]
        return recommendations
    else:
        return f"<p>Product {aid} not found!</p>", 404
    
@app.route("/api/v1/popular")
def popular():
    return ten_most_popular


if __name__ == '__main__':  
   print("Starting service...")
   app.run(debug = True)  