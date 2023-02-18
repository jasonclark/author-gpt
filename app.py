from flask import Flask, request, render_template, url_for
app = Flask(__name__)

from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
import torch.nn.functional as F
from random import choice

@app.route("/")
@app.route("/index/")
def index():
    return render_template('index.html', value='success')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/result/')
def result():
    return render_template('result.html')

def get_pred(text, model, tok, p=0.7):
    input_ids = torch.tensor(tok.encode(text)).unsqueeze(0)
    logits = model(input_ids, labels=input_ids)[1][:, -1]
    probs = F.softmax(logits, dim=-1).squeeze()
    idxs = torch.argsort(probs, descending=True)
    res, cumsum = [], 0.
    for idx in idxs:
        res.append(idx)
        cumsum += probs[idx]
        if cumsum > p:
            pred_idx = idxs.new_tensor([choice(res)])
            break
    pred = tok.convert_ids_to_tokens(int(pred_idx))
    return tok.convert_tokens_to_string(pred)

@app.route("/", methods=("GET","POST"))
@app.route("/index/", methods=("GET","POST"))
@app.route("/result/", methods=("GET","POST"))
def gpt_predictor(n=3):
    if request.method == 'POST':
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        text = request.form.get('text')
        n = request.form.get('n')
        for i in range(int(n)):
            pred = get_pred(text, model, tok)
            if pred == "<|endoftext|>":
                break
            else:
                text += pred
        return render_template('result.html',text = text)

if __name__ == '__main__':
    app.run(debug=True)
