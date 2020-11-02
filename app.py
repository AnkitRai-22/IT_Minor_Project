import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import torch
from sentence_transformers import SentenceTransformer,util

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

input = request.form['query']
def val2dict(input)
    embedder=SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    corpus=['boy','girl','india','fruit']
    corpus_embed = embedder.encode(corpus, convert_to_tensor=True)
    queries=[input]
    top_k=4
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embed)[0]
        cos_scores = cos_scores.cpu()
        top_results = torch.topk(cos_scores, k=top_k)

        for score, idx in zip(top_results[0], top_results[1]):
            output[corpus[idx]] = float(score)

        return render_template('index.html', prediction_text='Similar words are $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
