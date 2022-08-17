# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 12:25:37 2022

@author: Marie-Anne Melis
"""
import tweepy
from keys import *  #api keys etc
import numpy as np
import pandas as pd
import nltk
#import requests
import plotly.express as px 
#import matplotlib.pyplot as plt #Pyplot is used to generate the wordcloud plot
from wordcloud import WordCloud
from langdetect import detect
#import requests,json
from googletrans import Translator

import dash
from dash import dcc
from dash import html
import dash.dependencies as dd
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import base64  #imagecreation for wordcloud in dash app
from io import BytesIO #handling the ping image without saving it

#insert the keys you obtained from developer.twitter when creating
#your app.

client = tweepy.Client( bearer_token=bearer_token, 
                        consumer_key=consumer_key, 
                        consumer_secret=consumer_secret, 
                        access_token=access_token, 
                        access_token_secret=access_token_secret, 
                        wait_on_rate_limit=True)

# Define query

id=18171549 #hello twitter api, i wanna know about me


# no limit, try your best.

mytweets_arr_limitless = []

paginator = tweepy.Paginator(
    client.get_users_tweets,               # The method you want to use
    id,                            # Some argument for this method
    end_time="2022-03-31T11:59:59Z",       # Some argument for this method
    start_time="2010-11-06T11:59:59Z",     # Some argument for this method
    max_results=100,                        # How many tweets asked per request
    
)

try: 
    for tweet in paginator.flatten():      # Default to inf
       mytweets_arr_limitless = np.append(mytweets_arr_limitless, tweet.text)
except tweepy.RateLimitError as exc:
        print('Rate limit!')


#not much but all the tweettexts I could get (ca. 2800)

mytweets_flat = pd.DataFrame(mytweets_arr_limitless, columns=['text'])

#transformation
mytweets_flat['text'] = mytweets_flat['text'].astype(str).str.lower()

#remove all @, replies get the wrong language and why no other users in the
#wordcloud

mytweets_flat['text'] = mytweets_flat.apply(lambda x: x.str.replace('@',''))

#Detect language, since I now have installed non-buggy googletrans 3.0.1a I could
#use googletrans too.

mytweets_flat["language"] = None

for index, row in mytweets_flat.iterrows():
    #print(row['text'])
    try:
        detect_language = detect(row['text'])
        #print(detect_language)
    except:
        detect_language = "error"
        #print("This row throws and error:", row['text'])
    row['language'] = detect_language
    
 
#we only pick dutch and english tweets
mytweets_analyze = mytweets_flat.loc[mytweets_flat.language.isin(["nl", "en"]) ]


#to create an analysis with vader (only english) loop again and
#get translations for the text field. This is a good moment
#to do it because nothing has happened and the context
#is complete for the translation. I use googletrans because it's free?
#there are limits but far away for this experiment.

translator = Translator()

for index, row in mytweets_analyze.iterrows():

    from_lang=row["language"]
    if (from_lang == "en") or (from_lang == "en-US"):
        translation = row["text"]
        translator_name = "No translation needed"
    else:
        translation = translator.translate(row["text"], src='nl')
        row["text"]=translation.text

#create a csv in case that I exceed a limit or two @twitter or @google.
#the csv can be used instead of dynamically getting recent tweets
    
mytweets_analyze.to_csv("mytweets_analyze_translated.csv")   

    
#tokenization
from nltk.tokenize import RegexpTokenizer

regexp = RegexpTokenizer('\w+')

mytweets_analyze['text_token']=mytweets_analyze['text'].apply(regexp.tokenize)

#stopwoorden, maar ja, nederlandse stopwoorden

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

# Make a list of english and dutch stopwords, I could now remove dutch.
# because google has translated those texts into english

stopwords = nltk.corpus.stopwords.words("english")
stopwords_nl = nltk.corpus.stopwords.words("dutch")

# Extend the list with your own custom stopwords
my_stopwords = ['https','wel','graag','echt', 'weer', 'even','gewoon']
stopwords.extend(stopwords_nl)
stopwords.extend(my_stopwords)

#remove stopwords
mytweets_analyze['text_token'] = mytweets_analyze['text_token'].apply(lambda x: [item for item in x if item not in stopwords])

#remove infrequent words
mytweets_analyze['text_string'] = mytweets_analyze['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))

#Create a list of all words
all_words = ' '.join([word for word in mytweets_analyze['text_string']])


#Tokenize all_words
tokenized_words = nltk.tokenize.word_tokenize(all_words)


#Create a frequency distribution which records the number of times each word has occurred:
    
nltk.download('punkt')
from nltk.probability import FreqDist

fdist = FreqDist(tokenized_words)


#Now we can use our fdist dictionary to drop words which occur less than a certain amount of times (usually we use a value of 3 or 4).
#Since our dataset is really small, we don’t filter out any words and set the value to greater or equal to 1 (otherwise there are not many words left in this particular dataset)
mytweets_analyze['text_string_fdist'] = mytweets_analyze['text_token'].apply(lambda x: ' '.join([item for item in x if fdist[item] >= 1 ]))

#Next, we perfom lemmatization.

from nltk.stem import WordNetLemmatizer

wordnet_lem = WordNetLemmatizer()

mytweets_analyze['text_string_lem'] = mytweets_analyze['text_string_fdist'].apply(wordnet_lem.lemmatize)

# check if the columns are equal
mytweets_analyze['is_equal']= (mytweets_analyze['text_string_fdist']==mytweets_analyze['text_string_lem'])

# show level count
mytweets_analyze.is_equal.value_counts()
#wordcloud
all_words_lem = ' '.join([word for word in mytweets_analyze['text_string_lem']])

#WORDCLOUD PLOT GENERATION IS EMBEDDED IN THE FUNCTIONS TO CREATE
#PLOTS FOR DASH (see def section BELOW)


#SENTIMENT ANALISYS, VADER lexicon

nltk.download('vader_lexicon')

#Sentiment
#Sentiment Intensity Analyzer
#Initialize an object of SentimentIntensityAnalyzer with name “analyzer”:
from nltk.sentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

#Polarity scores, Use the polarity_scores method:
#try to get a translation from dutch to english and get a polarityscore

    
mytweets_analyze['polarity'] = mytweets_analyze['text_string_lem'].apply(lambda x: analyzer.polarity_scores(x))    


#Transform data , Change data structure
mytweets_analyze = pd.concat(
    [mytweets_analyze.drop([ 'polarity'], axis=1), 
     mytweets_analyze['polarity'].apply(pd.Series)], axis=1)
    
## Create new variable with sentiment "neutral," "positive" and "negative"
mytweets_analyze['sentiment'] = mytweets_analyze['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
    
#Analyze data, Tweet with highest positive sentiment
print(mytweets_analyze.loc[mytweets_analyze['compound'].idxmax()].values)   

# Tweet with highest negative sentiment 
# ...seems to be a case of wrong classification because of the word "deficit"
print(mytweets_analyze.loc[mytweets_analyze['compound'].idxmin()].values)
# Number of tweets 


########################OUTPUT###########################



def histogram_sentiment():
    #df = px.data.tips()
    histogram_sentiment = px.histogram(mytweets_analyze, x="sentiment", template="plotly_dark")
    return histogram_sentiment

def histogram_language():
    #df = px.data.tips()
    histogram_language = px.histogram(mytweets_analyze, x="language", template="plotly_dark")
    return histogram_language

def plot_wordcloud():
    #the callback function uses plot_wordcloud() to generate the actual 
    #graph and save it as a png. The png is displayed on the dashboard.
    
    wc = WordCloud(width=600, 
                         height=400, 
                         random_state=2, 
                         max_font_size=100).generate(all_words_lem)
    return wc.to_image()

#DASH LAYOUT

app=dash.Dash()
#load Cyborg and fontawesome
app = dash.Dash(external_stylesheets=[
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
     dbc.themes.CYBORG
])

app.layout = dbc.Container([
    dbc.Row([
        html.H3(children='My tweets analysis, subset english & dutch, dutch translated by google to english right after import.'),
        ]),
    dbc.Row([
        dbc.Col([      
                
                html.Div([
                    html.Img(id="image_wc"),
                    ]),
                dcc.Graph(id = 'histogram_sentiment',figure=histogram_sentiment()),
                ],width=6),

            dbc.Col([      
               dcc.Graph(id = 'histogram_language',figure=histogram_language()),

                ],width=6),
            ]),  
        ])


@app.callback(dd.Output('image_wc', 'src'), [dd.Input('image_wc', 'id')])
def make_image(b):
    img = BytesIO()
    plot_wordcloud().save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

if __name__== '__main__':
    app.run_server()