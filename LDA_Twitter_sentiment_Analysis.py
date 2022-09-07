#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df1= pd.read_csv(r"C:\Users\anushka\Desktop\New folder\Desktop\EXTRACTED\combined\csv\sales1.csv")


# In[3]:


df2=pd.read_csv(r"C:\Users\anushka\Desktop\New folder\Desktop\EXTRACTED\combined\csv\sales2.csv")


# In[4]:


df3=pd.read_csv(r"C:\Users\anushka\Desktop\New folder\Desktop\EXTRACTED\combined\csv\sales3.csv")


# In[5]:


frames=[df1,df2,df3]


# In[6]:


result=pd.concat(frames)


# In[7]:


result


# In[8]:


result1 = result.rename({'Data.Column2':'Tweets'}, axis=1)
result1


# In[9]:


get_ipython().system('pip install wordcloud')


# In[11]:


get_ipython().system('pip install langdetect')
get_ipython().system('pip install pycountry')
get_ipython().system('pip install textblob')
get_ipython().system('pip install tweepy')

from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
import re
from nltk.tokenize import word_tokenize
import seaborn as sns


#reading csv file containing scrapped tweets:
tweet_list = result1

#dropping duplicate tweets from Tweets column:

tweet_list.drop_duplicates(subset=['Tweets'] ,keep = False, inplace = True)


# In[12]:


def percentage(part,whole):
     return 100 * float(part)/float(whole)
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#function to cleaning the tweets
def clean_tweet(Tweets):
    if type(Tweets) == np.float:
        return ""
    temp = Tweets.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    temp = [w for w in temp if not w in stop_words]
    temp = " ".join(temp)
    temp = temp.replace('RT','')
    temp = temp.replace('rt','')
    return temp

#function to keep only first 100 characters of the tweets
def shorten_tweets(Tweets):
    temp = Tweets[0:100]
    return temp
ID_TWEET = pd.DataFrame(result1)
ID_TWEET['clean_tweets'] = ID_TWEET['Tweets']
#Removing RT, Punctuation etc:
ID_TWEET['clean_tweets'] = ID_TWEET['clean_tweets'].apply(clean_tweet)
ID_TWEET['clean_tweets'] = ID_TWEET['clean_tweets'].apply(clean_tweet)

#adding new column to the dataframe and checking for duplicates:
ID_TWEET['first100charactersoftweets'] =ID_TWEET['clean_tweets'].apply(shorten_tweets)
ID_TWEET = ID_TWEET.drop_duplicates(subset='first100charactersoftweets', keep="first")


# In[13]:


ID_TWEET


# In[14]:


ID_TWEET=ID_TWEET.drop(["Data.Column1","Tweets","Name"],axis=1)


# In[15]:


from textblob import TextBlob
# function to calculate subjectivity
def getSubjectivity(clean_tweets):
    return TextBlob(clean_tweets).sentiment.subjectivity
    # function to calculate polarity
def getPolarity(clean_tweets):
    return TextBlob(clean_tweets).sentiment.polarity

# function to analyze the reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
fin_data = pd.DataFrame(ID_TWEET)
fin_data['Polarity'] = fin_data['clean_tweets'].apply(getPolarity) 
fin_data['Analysis'] = fin_data['Polarity'].apply(analysis)
fin_data_final=fin_data.drop(["first100charactersoftweets"],axis=1)

#create new data frames for all sentiments
tweet_neg = fin_data[fin_data["Analysis"] == "Negative"]
tweet_neu = fin_data[fin_data["Analysis"] == "Neutral"]
tweet_pos = fin_data[fin_data["Analysis"] == "Positive"]
#function for calculating the percentage of all the sentiments
def calc_percentage(x,y):
    return x/y * 100
pos_per = calc_percentage(len(tweet_pos), len(fin_data_final))
neg_per = calc_percentage(len(tweet_neg), len(fin_data_final))
neu_per = calc_percentage(len(tweet_neu), len(fin_data_final))
print("positive: {} {}%".format(len(tweet_pos),  format(pos_per, '.1f')))
print("negative: {} {}%".format(len(tweet_neg), format(neg_per, '.1f')))
print("neutral: {} {}%".format(len(tweet_neu), format(neu_per, '.1f')))


# In[16]:


fin_data_final


# In[17]:


fin_data_final.to_csv(r"C:\Users\anushka\Desktop\TextblobNew.csv")


# In[18]:


ID_TWEET=pd.DataFrame(result1)


# In[19]:


ID_TWEET


# In[20]:


import nltk
import pandas as pd

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
for index, row in ID_TWEET['Tweets'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    if score['neg'] > score['pos']:
        ID_TWEET.loc[index, "Sentiment"] = "negative"
    elif score['pos'] > score['neg']:
        ID_TWEET.loc[index, "Sentiment"] = "positive"
    else:
        ID_TWEET.loc[index, "Sentiment"] = "neutral"
        
    ID_TWEET.loc[index, 'neg'] = score['neg']
    ID_TWEET.loc[index, 'neu'] = score['neu']
    ID_TWEET.loc[index, 'pos'] = score['pos']
    ID_TWEET.loc[index, 'compound'] = score['compound']
VADER_SENTIMENT= pd.DataFrame(ID_TWEET)
VADER_SENTIMENT= pd.DataFrame(ID_TWEET[['Tweets','neg','neu','pos','compound']])

# function to analyze the reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
VADER_SENTIMENT['Analysis'] = VADER_SENTIMENT['compound'].apply(analysis)

#create new data frames for all sentiments
tweet_neg = VADER_SENTIMENT[VADER_SENTIMENT["Analysis"] == "Negative"]
tweet_neu = VADER_SENTIMENT[VADER_SENTIMENT["Analysis"] == "Neutral"]
tweet_pos = VADER_SENTIMENT[VADER_SENTIMENT["Analysis"] == "Positive"]
#function for calculating the percentage of all the sentiments
def calc_percentage(x,y):
    return x/y * 100
pos_per = calc_percentage(len(tweet_pos), len(VADER_SENTIMENT))
neg_per = calc_percentage(len(tweet_neg), len(VADER_SENTIMENT))
neu_per = calc_percentage(len(tweet_neu), len(VADER_SENTIMENT))
print("positive: {} {}%".format(len(tweet_pos),  format(pos_per, '.1f')))
print("negative: {} {}%".format(len(tweet_neg), format(neg_per, '.1f')))
print("neutral: {} {}%".format(len(tweet_neu), format(neu_per, '.1f')))


# In[21]:


VADER_SENTIMENT


# In[22]:


import pandas as pd
import io
from io import StringIO
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_csv(r'C:\Users\anushka\Desktop\TextblobNew.csv') 
data


# In[23]:


data=data.loc[:, ~data.columns.str.contains('^Unnamed')]
data


# In[29]:


X = data['clean_tweets']
y = data['Analysis']
X = X.values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=105)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[30]:


data = data.dropna()
data


# In[31]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for feat in ['clean_tweets','Analysis']:
    data[feat]=le.fit_transform(data[feat].astype(str))
data


# In[32]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

sc = StandardScaler()
sc.fit_transform(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svc = SVC(kernel='linear', C=10.0, random_state=1)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[33]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =1
                          ).fit(X_train, y_train)
 

accuracy = knn.score(X_test, y_test)
print (accuracy)
 

knn_predictions = knn.predict(X_test)
cm = confusion_matrix(y_test, knn_predictions)
print("Precision Score : ",precision_score(y_test, y_pred, pos_label='positive',average='micro'))
print("Recall Score : ",recall_score(y_test, y_pred,pos_label='positive',average='micro'))

print(cm)
predictions =knn.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[34]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators =1)

clf.fit(X_train, y_train)
  

random_Forest = clf.predict(X_test)
cm = confusion_matrix(y_test, random_Forest)
y_pred = clf.predict(X_test)
print(cm)

from sklearn import metrics 
print()
 

print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
print("Precision Score : ",precision_score(y_test, y_pred, pos_label='positive',average='micro'))
                                           
print("Recall Score : ",recall_score(y_test, y_pred, pos_label='positive',average='micro'))
predictions =clf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[35]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)
 

accuracy = gnb.score(X_test, y_test)
print (accuracy)
 

cm = confusion_matrix(y_test, gnb_predictions)
print(cm)
print("Precision Score : ",precision_score(y_test, y_pred, pos_label='positive',average='micro'))
                                           
print("Recall Score : ",recall_score(y_test, y_pred, pos_label='positive',average='micro'))
predictions =gnb.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[36]:


X.shape


# In[37]:


X.reshape(-1,1)


# In[38]:


from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
 


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =0)
 
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
accuracy = dtree_model.score(X_test, y_test)
print(accuracy)
 

cm = confusion_matrix(y_test, dtree_predictions)
print(cm)

                                           

predictions = dtree_model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[39]:



import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
my_model = xgb.XGBClassifier()
my_model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = my_model.predict(X_test)
accuracy = my_model.score(X_test, y_test)
print(accuracy)
   
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Precision Score : ",precision_score(y_test, y_pred, pos_label='positive',average='micro'))
                                           
print("Recall Score : ",recall_score(y_test, y_pred, pos_label='positive',average='micro'))
print(cm)

predictions =my_model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[40]:


import pandas as pd
import io
from io import StringIO
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_csv(r'C:\Users\anushka\Desktop\VADER_SENTIMENT.csv') 
data


# In[41]:


data=data.loc[:, ~data.columns.str.contains('^Unnamed')]
data


# In[46]:


X = data['Tweets']
y = data['Analysis']
X = X.values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=105)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[47]:


data = data.dropna()
data


# In[48]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for feat in ['Tweets','Analysis']:
    data[feat]=le.fit_transform(data[feat].astype(str))
data


# In[49]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
#
# Standardize the data set
#
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#
# Fit the SVC model
#
svc = SVC(kernel='linear', C=10.0, random_state=1)
svc.fit(X_train, y_train)
#
# Get the predictions
#
y_pred = svc.predict(X_test)
#
# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[50]:


df=pd.DataFrame(ID_TWEET[['clean_tweets']])
df


# In[51]:


import pandas as pd
import numpy as np

import string
import math

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from wordcloud import WordCloud

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
lemmatizer = WordNetLemmatizer()

def preprocess(df_clean_tweets):
    tokens = word_tokenize(df_clean_tweets)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words and len(token) > 3]
    
    lemmatized =[]
    
    for w in stopwords_removed:
        lemmatized.append(lemmatizer.lemmatize(w))
        
    processed = list(filter(lambda x: x.isalpha(), lemmatized))  
        
    return processed
df['clean_tweets'] = df['clean_tweets'].apply(preprocess)


# In[52]:


all_words = [word for tokens in df['clean_tweets'] for word in tokens]
tweet_lengths = [len(tokens) for tokens in df['clean_tweets']]
vocab = sorted(list(set(all_words)))

print('{} tokens total, with a vocabulary size of {}'.format(len(all_words), len(vocab)))
print('Max tweet length is {}'.format(max(tweet_lengths)))


# In[53]:


word_length = []
for word in all_words:
    word_length.append(len(word))


# In[54]:


print('average word size is {}'.format( sum(word_length) / len(word_length)))


# In[55]:


df


# In[56]:


text_dict = Dictionary(df.clean_tweets)


# In[57]:


text_dict.filter_extremes(no_below = 5, no_above = .90)


# In[58]:


txt_out = text_dict.token2id


# In[59]:


tweets_bow = [text_dict.doc2bow(tweet) for tweet in df['clean_tweets']]


# In[60]:


df['clean_tweets'][760]


# In[61]:


tweets_bow[0]


# In[62]:


k = 5
tweets_lda = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)


# In[63]:


tweets_lda.show_topics()


# In[64]:


df = df.loc[~df.index.duplicated(keep='first')]
def format_topics_sentences(ldamodel=None, corpus=tweets_bow, texts=df['clean_tweets']):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[tweets_bow]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=tweets_lda, corpus=tweets_bow, texts=df['clean_tweets'])

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)


# In[65]:


df_dominant_topic.head(20)


# In[66]:


def plot_top_words(lda=tweets_lda, nb_topics=k, nb_words=10):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = gridspec.GridSpec(round(math.sqrt(k))+1,round(math.sqrt(k))+1)
    gs.update(wspace=.5, hspace=.5)
    plt.figure(figsize=(30,20))
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='blue', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Topic "+str(i))
        
plot_top_words()


# In[67]:


import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()

pyLDAvis.enable_notebook()

# feed the LDA model into the pyLDAvis instance
lda_viz = gensimvis.prepare(tweets_lda,tweets_bow, dictionary=tweets_lda.id2word)
lda_viz

