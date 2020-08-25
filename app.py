import model
import traceback
import sys

from flask import request
from flask import Flask
from flask import jsonify
from sklearn.externals import joblib


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predicts():
    if model:
        try:
            #text_ = request.txt
            #context = pd.get_dummies(pd.DataFrame(text_))
            context = "New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean. It has a total land area of 268,000 square kilometres (103,500 sq mi), and a population of 4.9 million. New Zealand's capital city is Wellington, and its most populous city is Auckland."
            questions = ['How many people live in New Zealand?']
            predictions = model.run_prediction(questions, context)
            return jsonify({'prediction': predictions})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return 'No model here to use'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 5000
    model = joblib.load('./model')
    print('Model loaded')
    app.run(host='127.0.0.1', port=port, debug=True)
