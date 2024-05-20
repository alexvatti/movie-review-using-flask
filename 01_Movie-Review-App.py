import numpy as np
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from flask import Flask, request, render_template

app=Flask(__name__)
ml_model = pickle.load(open('movie_review_rbf_model.pkl', 'rb'))
vectorizer_model = pickle.load(open('count_vectorizer_movie_review.pkl','rb'))


def pre_processing(text):
    words =  word_tokenize(text) 
    eng_stopwords=stopwords.words("english")
    tokens = [word for word in words if word.isalnum()]
    lower_tokens = [word.lower() for word in tokens ]

    no_punctuations_stopwords_tokens = [token for  token in lower_tokens if token not in eng_stopwords]
    return " ".join(no_punctuations_stopwords_tokens)

@app.route('/')      # decorator
def home():
      review_image = 'static/sentiment.jpeg'
      return render_template('index.html',review_image=review_image)
    
    
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    #text = request.form['Movie Review']
    text=list(request.form.values())[0]
    print(text)
    processed_text = pre_processing(text)
    X = vectorizer_model.transform([processed_text])
    prediction = ml_model.predict(X)

    ans =prediction[0]
    print(ans)
    if ans == 1: 
        review='Positive' 
    else:
        review='Negative'
    review_image = 'static/sentiment.jpeg'
    return render_template('index.html', review_image=review_image,prediction_text='Moview Review Classified As {} '.format(review))


if __name__=='__main__':
    app.run(debug=True)
