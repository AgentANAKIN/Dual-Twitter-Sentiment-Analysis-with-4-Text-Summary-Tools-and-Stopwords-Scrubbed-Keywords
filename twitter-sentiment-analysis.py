# acknowledgements
# https://github.com/llSourcell/twitter_sentiment_challenge/blob/master/demo.py
# https://youtu.be/qTyj2R-wcks
# https://www.youtube.com/watch?v=8p9nHmtwk0o
# https://github.com/Jcharis/Natural-Language-Processing-Tutorials/blob/master/Text%20Summarization%20with%20Sumy%20Python%20.ipynb
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
# https://codeburst.io/python-basics-11-word-count-filter-out-punctuation-dictionary-manipulation-and-sorting-lists-3f6c55420855
# https://pythonspot.com/nltk-stop-words/



# import dependency: Twitter API
import tweepy
# import dependencies: sentiment analysis
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import dependency: pie chart
import matplotlib.pyplot as plt
# import dependencies: text summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk; nltk.download('punkt')
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
# import dependencies: keywords
import nltk; nltk.download('stopwords')
from nltk.corpus import stopwords



# improves performance
SIA = SentimentIntensityAnalyzer()



# access Twitter's API
consumer_key= 'CONSUMER_KEY_HERE'
consumer_secret= 'CONSUMER_SECRET_HERE'
access_token='ACCESS_TOKEN_HERE'
access_token_secret='ACCESS_TOKEN_SECRET_HERE'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)



# extended mode reads up to 280 characters 
num = 100
#tweets = tweet_batch = api.search(q='chocolate', lang='en', tweet_mode='extended', count=num)
tweets = tweet_batch = api.search(q=input('Search term: '), lang='en', tweet_mode='extended', count=num)



# set some variables
_max_queries = 100  
ct = 1
negative = 0
positive = 0
neutral = 0
unknown = 0



# create text files to append tweets to
txtNegative = open('negative.txt','a+')
txtPositive = open('positive.txt','a+')
txtNeutral = open('neutral.txt','a+')
txtUnknown = open('unknown.txt','a+')



# classify tweets as negative, positive, neutral, or unknown
# TextBlob and VADER must agree, or the result is "unknown"
while len(tweets) < num and ct < _max_queries:
    for tweet in tweets:
        #print(tweet.full_text)
        analysisTB = TextBlob(tweet.full_text)
        analysisVS = SIA.polarity_scores(tweet.full_text)
        if ((analysisTB.sentiment.polarity < -0.05) & 
            (analysisVS['compound'] < -0.05)):
            txtNegative.write(tweet.full_text)
            negative += 1
        elif ((analysisTB.sentiment.polarity > 0.05) & 
              (analysisVS['compound'] > 0.05)):
            txtPositive.write(tweet.full_text)
            positive += 1
        elif ((analysisTB.sentiment.polarity > -0.05) & 
              (analysisTB.sentiment.polarity < 0.05) & (analysisVS['compound'] > -0.05) & (analysisVS['compound'] < 0.05)):
            txtNeutral.write(tweet.full_text)
            neutral += 1
        else:
            txtUnknown.write(tweet.full_text)
            unknown += 1
        #print(analysisTB.sentiment)
        #print(analysisVS)
        #print("") 
    ct += 1



# open file to append summaries to
txtSummary = open('summary.txt','a+')



# print totals on screen and to file
total = negative + positive + neutral + unknown
negative_pct = ((negative / total) * 100)
txtSummary.write("Negative: ")
txtSummary.write(str(negative))
txtSummary.write(" ")
txtSummary.write(str(negative_pct))
txtSummary.write("%")
print("Negative: "+str(negative))

positive_pct = ((positive / total) * 100)
txtSummary.write("\n\nPositive: ")
txtSummary.write(str(positive))
txtSummary.write(" ")
txtSummary.write(str(positive_pct))
txtSummary.write("%")
print("Positive: "+str(positive))

neutral_pct = ((neutral / total) * 100)
txtSummary.write("\n\nNeutral: ")
txtSummary.write(str(neutral))
txtSummary.write(" ")
txtSummary.write(str(neutral_pct))
txtSummary.write("%")
print("Neutral: "+str(neutral))

unknown_pct = ((unknown / total) * 100)
txtSummary.write("\n\nUnknown: ")
txtSummary.write(str(unknown))
txtSummary.write(" ")
txtSummary.write(str(unknown_pct))
txtSummary.write("%")
print("Unknown: "+str(unknown))



# displays the results as a pie chart and explodes the largest slice
labels = ['negative', 'positive', 'neutral', 'unknown']
sizes = [negative, positive, neutral, unknown]
colors = ['red', 'green', 'yellow', 'gray']
if ((negative > positive) & 
    (negative > neutral) & 
    (negative > unknown)):
    explode = [.1, 0, 0, 0]
elif ((positive > negative) & 
      (positive > neutral) & 
      (positive > unknown)):
    explode = [0, .1, 0, 0]
elif ((neutral > negative) & 
      (neutral > positive) & 
      (neutral > unknown)):
    explode = [0, 0, .1, 0]
else:
    explode = [0, 0, 0, .1]
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()



# close files and reopen for reading
txtNegative.close()
txtPositive.close()
txtNeutral.close()
txtUnknown.close()
txtNegative = open('negative.txt','r')
txtPositive = open('positive.txt','r')
txtNeutral = open('neutral.txt','r')
txtUnknown = open('unknown.txt','r')



# stop words are common English words, such as "the;" the lists of keywords will exclude these stop words
stopWords = set(stopwords.words('english'))



# summarize tweets with LexRank, Luhn, LSA, Stop Words
parser = PlaintextParser.from_file("negative.txt", Tokenizer("english"))
LRSummarizer = LexRankSummarizer()
summary = LRSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LEXRANK NEGATIVE ***\n")
print("*** LEXRANK NEGATIVE ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSummarizer = LuhnSummarizer()
summary = LSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LUHN NEGATIVE ***\n")
print("")
print("*** LUHN NEGATIVE ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSASummarizer = LsaSummarizer()
summary = LSASummarizer(parser.document, 1)
txtSummary.write("\n\n*** LSA NEGATIVE ***\n")
print("")
print("*** LSA NEGATIVE ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSA2Summarizer = LsaSummarizer()
LSA2Summarizer = LsaSummarizer(Stemmer("english"))
LSA2Summarizer.stop_words = get_stop_words("english")
txtSummary.write("\n\n*** STOP WORDS NEGATIVE ***\n")
print("")
print("*** STOP WORDS NEGATIVE ***")
for sentence in LSA2Summarizer(parser.document, 1):
    txtSummary.write(str(sentence))
    print(sentence)

# clean text and convert all words to lowercase
Text = open('negative.txt').read()
for char in '-.,:;?!\n':
    Text = Text.replace(char,' ')
Text = Text.lower()
word_list = Text.split()
# initialize dictionary
d = {}
# count instances of each word
for word in word_list:
    if word not in d:
        d[word] = 0
    d[word] += 1
# reverse the key and values so they can be sorted using tuples
# discard common words and words that appear only once
word_freq = []
for key, value in d.items():
    if (value > 1) and (key not in stopWords):
        word_freq.append((value, key))
word_freq.sort(key=lambda tup:(-tup[0], tup[1]))
txtSummary.write("\n\n*** NEGATIVE KEYWORDS ***\n")
print("")
print("*** NEGATIVE KEYWORDS ***")
for word in word_freq:
    txtSummary.write(str(word))
    txtSummary.write("\n")
    print(word)

parser = PlaintextParser.from_file("positive.txt", Tokenizer("english"))
LRSummarizer = LexRankSummarizer()
summary = LRSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LEXRANK POSITIVE ***\n")
print("")
print("*** LEXRANK POSITIVE ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSummarizer = LuhnSummarizer()
summary = LSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LUHN POSITIVE ***\n")
print("")
print("*** LUHN POSITIVE ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSASummarizer = LsaSummarizer()
summary = LSASummarizer(parser.document, 1)
txtSummary.write("\n\n*** LSA POSITIVE ***\n")
print("")
print("*** LSA POSITIVE ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSA2Summarizer = LsaSummarizer()
LSA2Summarizer = LsaSummarizer(Stemmer("english"))
LSA2Summarizer.stop_words = get_stop_words("english")
txtSummary.write("\n\n*** STOP WORDS POSITIVE ***\n")
print("")
print("*** STOP WORDS POSITIVE ***")
for sentence in LSA2Summarizer(parser.document, 1):
    txtSummary.write(str(sentence))
    print(sentence)

# clean text and convert all words to lowercase
Text = open('positive.txt').read()
for char in '-.,:;?!\n':
    Text = Text.replace(char,' ')
Text = Text.lower()
word_list = Text.split()
# initialize dictionary
d = {}
# count instances of each word
for word in word_list:
    if word not in d:
        d[word] = 0
    d[word] += 1
# reverse the key and values so they can be sorted using tuples
# discard common words and words that appear only once
word_freq = []
for key, value in d.items():
    if (value > 1) and (key not in stopWords):
        word_freq.append((value, key))
word_freq.sort(key=lambda tup:(-tup[0], tup[1]))
txtSummary.write("\n\n*** POSITIVE KEYWORDS ***\n")
print("")
print("*** POSITIVE KEYWORDS ***")
for word in word_freq:
    txtSummary.write(str(word))
    txtSummary.write("\n")
    print(word)

parser = PlaintextParser.from_file("neutral.txt", Tokenizer("english"))
LRSummarizer = LexRankSummarizer()
summary = LRSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LEXRANK NEUTRAL  ***\n")
print("")
print("*** LEXRANK NEUTRAL ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSummarizer = LuhnSummarizer()
summary = LSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LUHN NEUTRAL  ***\n")
print("")
print("*** LUHN NEUTRAL ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSASummarizer = LsaSummarizer()
summary = LSASummarizer(parser.document, 1)
txtSummary.write("\n\n*** LSA NEUTRAL  ***\n")
print("")
print("*** LSA NEUTRAL ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSA2Summarizer = LsaSummarizer()
LSA2Summarizer = LsaSummarizer(Stemmer("english"))
LSA2Summarizer.stop_words = get_stop_words("english")
txtSummary.write("\n\n*** STOP WORDS NEUTRAL  ***\n")
print("")
print("*** STOP WORDS NEUTRAL ***")
for sentence in LSA2Summarizer(parser.document, 1):
    txtSummary.write(str(sentence))
    print(sentence)

# clean text and convert all words to lowercase
Text = open('neutral.txt').read()
for char in '-.,:;?!\n':
    Text = Text.replace(char,' ')
Text = Text.lower()
word_list = Text.split()
# initialize dictionary
d = {}
# count instances of each word
for word in word_list:
    if word not in d:
        d[word] = 0
    d[word] += 1
# reverse the key and values so they can be sorted using tuples
# discard common words and words that appear only once
word_freq = []
for key, value in d.items():
    if (value > 1) and (key not in stopWords):
        word_freq.append((value, key))
word_freq.sort(key=lambda tup:(-tup[0], tup[1]))
txtSummary.write("\n\n*** NEUTRAL KEYWORDS ***\n")
print("")
print("*** NEUTRAL KEYWORDS ***")
for word in word_freq:
    txtSummary.write(str(word))
    txtSummary.write("\n")
    print(word)

parser = PlaintextParser.from_file("unknown.txt", Tokenizer("english"))
LRSummarizer = LexRankSummarizer()
summary = LRSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LEXRANK UNKNOWN  ***\n")
print("")
print("*** LEXRANK UNKNOWN ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSummarizer = LuhnSummarizer()
summary = LSummarizer(parser.document, 1)
txtSummary.write("\n\n*** LUHN UNKNOWN  ***\n")
print("")
print("*** LUHN UNKNOWN ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSASummarizer = LsaSummarizer()
summary = LSASummarizer(parser.document, 1)
txtSummary.write("\n\n*** LSA UNKNOWN  ***\n")
print("")
print("*** LSA UNKNOWN ***")
for sentence in summary:
    txtSummary.write(str(sentence))
    print(sentence)

LSA2Summarizer = LsaSummarizer()
LSA2Summarizer = LsaSummarizer(Stemmer("english"))
LSA2Summarizer.stop_words = get_stop_words("english")
txtSummary.write("\n\n*** STOP WORDS UNKNOWN  ***\n")
print("")
print("*** STOP WORDS UNKNOWN ***")
for sentence in LSA2Summarizer(parser.document, 1):
    txtSummary.write(str(sentence))
    print(sentence)

# clean text and convert all words to lowercase
Text = open('unknown.txt').read()
for char in '-.,:;?!\n':
    Text = Text.replace(char,' ')
Text = Text.lower()
word_list = Text.split()
# initialize dictionary
d = {}
# count instances of each word
for word in word_list:
    if word not in d:
        d[word] = 0
    d[word] += 1
# reverse the key and values so they can be sorted using tuples
# discard common words and words that appear only once
word_freq = []
for key, value in d.items():
    if (value > 1) and (key not in stopWords):
        word_freq.append((value, key))
word_freq.sort(key=lambda tup:(-tup[0], tup[1]))
txtSummary.write("\n\n*** UNKNOWN KEYWORDS ***\n")
print("")
print("*** UNKNOWN KEYWORDS ***")
for word in word_freq:
    txtSummary.write(str(word))
    txtSummary.write("\n")
    print(word)



# close files
txtNegative.close()
txtPositive.close()
txtNeutral.close()
txtUnknown.close()
txtSummary.close()
