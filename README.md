# Twitter Sentiment Analysis Pipeline for E-Health Data
This repository contains the code and resources for an end-to-end Twitter sentiment analysis pipeline designed to uncover insights, trends, and public sentiment related to E-Health using Twitter data.

**Overview:**
This project focuses on conducting sentiment analysis on Twitter data related to E-Health using specific hashtags. By leveraging this pipeline, you can extract tweets from Twitter's API, preprocess the data, visualize key insights, classify or tag tweets using textblob and Vader classifiers, and even perform N-grams analysis.

**Pipeline Components:**
The pipeline includes the following key components:

Tweet Extraction: We extract tweets from Twitter's API using specified hashtags. The code for this data scraping process is provided in this repository.

Data Preprocessing: The extracted data undergoes preprocessing, which includes text cleaning, tokenization, and removing stopwords. This step ensures the data is ready for analysis.

Data Visualization: We use data visualization techniques to create informative plots and charts that provide insights into the sentiment and trends in the E-Health data.

Classification/Tagging: Tweets are classified or tagged using two popular sentiment analysis classifiers: textblob and Vader. This step helps categorize tweets as positive, negative, or neutral.

Machine Learning: You have the option to train machine learning algorithms on the labeled data to improve sentiment classification. Feel free to explore different ML models to enhance accuracy.

N-grams Analysis: Analyze the data using N-grams, which can reveal important phrases and combinations of words that contribute to the sentiment of the tweets.

**Dataset Generation:** 
The dataset used in this project was generated by scraping tweets biweekly from Twitter. The code for data scraping is provided in this repository, allowing you to reproduce the dataset or customize it to your needs.

**Conclusion**: 
The extracted tweets were an amalgamation of words,emoticons,hashtags,links and symbols. These were cleaned using pre-processing techniques to make it suitable for feeding into the models. The dataset was labelled using Vader and textblob classifier and subsequently fed into machine learning algorithms like Naïve Bayes, Decision Tree,KNN classifier, Random Forest and XGBoost. Our finding suggests that people in India have been reading/posting about e-health on twitter Since negative category tweets were the least the over debate about e-health seems to be positive. The best accuracy was found for VADER classifier since its dictionary has a lot of words that are usually used online on social media sites, especially Twitter. 

