from flask import Flask, render_template, request
from main import predict_sentiment

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse', methods=['POST'])
def predict():
    text = request.form['text']
    sentiment = predict_sentiment(text)
    return render_template('index.html', sentiment=sentiment, text=text)

if __name__ == '__main__':
    app.run(debug=True)
