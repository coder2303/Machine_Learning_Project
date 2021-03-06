import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

#twitter implementation
import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json


def most_element(liste):
    numeral=[[liste.count(nb), nb] for nb in liste]
    numeral.sort(key=lambda x:x[0], reverse=True)
    return(numeral[0][1])

# choose most voted resukt for particular input  
class votes_classifier(ClassifierI):
  def __init__(self, *classifiers):
    self._classifiers=classifiers
    
  def classify(self,features):
    votes=[]
    for c in self._classifiers:
      v=c.classify(features)
      votes.append(v)
    return most_element(votes)
  
  def conf(self,features):
    votes=[]
    for c in self._classifiers:
      v=c.classify(features)
      votes.append(v)
    
    count=votes.count(most_element(votes))
    return count/len(votes)

# store text as either positive or negative
document_f = open("documents.pickle", "rb")
document_n = pickle.load(document_f)
document_f.close()
  
word_feature_f = open("word_features5k.pickle", "rb")
word_feature_n = pickle.load(word_feature_f)
word_feature_f.close()  

def find_features(document):
  word_list=word_tokenize(document)
  features={}
  for w in word_feature_n:
    features[w]=(w in word_list)
  
  return features

# restoring the features stored in pickle
feature_set_f = open("feature_set.pickle", "rb")
feature_set = pickle.load(feature_set_f)
feature_set_f.close()

random.shuffle(feature_set)

# positive data example:      
training_set = feature_set[:10000]
testing_set =  feature_set[10000:]

# Opening all the train classifier model in pickle
open_file = open("originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open("SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()



# return classifier with most voted result
voted_classifier = votes_classifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


# return the sentiment of a given tweet
def sentiment(text):
  feats=find_features(text)
  return voted_classifier.classify(feats),voted_classifier.conf(feats)

#consumer key, consumer secret, access token, access secret.
ckey="7dAfUqelYTF0vlwPDtcvDFpM1"
csecret="QMqSw2VzpJuAouibJQGsbkbyuUxyCHm8HMdHGK1EfJxYEIj85u"
atoken="1054118892707241984-BdPhnD1GCqbm2gjnlHoCFDUjrl0xHm"
asecret="I0naAWrvZT8pXoyBBXcHuyAdeDg2eKG0USGi9B8sWtiTn"


#working with twitter API
class listener(StreamListener):

    def on_data(self, data):

      all_data = json.loads(data)

      tweet = all_data["text"]
      sentiment_value, confidence = sentiment(tweet)
      print(tweet, sentiment_value, confidence)
      if confidence*100 >= 80:
            output = open("drive/My Drive/twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()
      return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["modi"])

# graph plot

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

style.use("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pullData = open("drive/My Drive/twitter-out.txt","r").read()
    lines = pullData.split('\n')

    xar = []
    yar = []

    x = 0
    y = 0

    for l in lines:
        x += 1
        if "pos" in l:
            y += 1
        elif "neg" in l:
            y -= 1

        xar.append(x)
        yar.append(y)
        
    ax1.clear()
    ax1.plot(xar,yar)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
