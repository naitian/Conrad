import pickle
from dataset import get_input_vector

from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


mr = pickle.load(open('./models/random_forest_model.normed.pkl', 'rb'))


@app.route('/', methods=['POST'])
def hello_world():
    sentence = request.get_json()['sentence']
    iv = get_input_vector(sentence)
    react_probs = mr.predict(iv.reshape(1, -1))
    print(react_probs.shape)
    print(react_probs)
    return jsonify({
        'love': int(react_probs[0][0] * 50),
        'haha': int(react_probs[0][1] * 50),
        'wow': int(react_probs[0][2] * 50),
        'sad': int(react_probs[0][3] * 50),
        'angry': int(react_probs[0][4] * 50),
    })
